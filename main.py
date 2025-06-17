#!/usr/bin/env python3
"""
MCP 기반 통합 프롬프트 주입 공격 프레임워크
논문 설계: 템플릿 선택 → 초기 공격 → 실패시 Sequential Thinking 개선 → 재시도 (최대 10번)
"""

import argparse
import asyncio
import os
import sys
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

# 환경 변수 로딩
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# MCP 관련 import
from mcp_manager import initialize_mcp_client, cleanup_mcp_client

# dataset 관련 import
from utils.run import init_dataset

# 새로운 통합 공격 프레임워크 import
from attack_framework.unified_attack_executor import UnifiedAttackExecutor, UnifiedAttackResult
from attack_framework.attack_templates import AttackCategory

class UnifiedAttackFramework:
    """통합 프롬프트 주입 공격 프레임워크"""
    
    def __init__(self, args):
        self.args = args
        self.mcp_client = None
        self.mcp_tools = None
        self.attack_executor = UnifiedAttackExecutor()
        
        # 환경변수 로드
        self.attacker_model = args.attacker_model or os.getenv("ATTACKER_MODEL", "qwen/qwen3-4b")
        self.target_model = args.target_model or os.getenv("TARGET_MODEL", "llama-3.2-1b-instruct")
        self.lm_studio_url = args.lm_studio_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.lm_studio_key = args.lm_studio_key or os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        
    async def initialize(self):
        """프레임워크 초기화"""
        print("🎯 MCP 기반 통합 프롬프트 주입 공격 프레임워크")
        print("=" * 60)
        print(f"공격자 LLM: {self.attacker_model}")
        print(f"피공격자 LLM: {self.target_model}")
        print(f"LM Studio URL: {self.lm_studio_url}")
        print("=" * 60)
        
        init_dataset()

        # MCP 클라이언트 초기화
        print("\n[1] MCP 클라이언트 초기화 중...")
        try:
            self.mcp_client, self.mcp_tools = await initialize_mcp_client()
            if not self.mcp_tools:
                raise Exception("MCP 도구를 찾을 수 없습니다")
            print(f"✓ {len(self.mcp_tools)} 개의 MCP 도구 로드됨")
            
            # 도구 목록 출력 및 공격자 LLM 확인
            attacker_llm_found = False
            for tool in self.mcp_tools:
                print(f"  - {tool.name}")
                if "chat_completion" in tool.name:
                    attacker_llm_found = True
            
            if not attacker_llm_found:
                print("⚠️  공격자 LLM (chat_completion) 도구를 찾을 수 없습니다!")
                print("   MCP 서버 mcp-pi.py가 실행 중인지 확인하세요.")
            else:
                print("✓ 공격자 LLM 도구 준비완료")
                
        except Exception as e:
            print(f"✗ MCP 초기화 실패: {e}")
            return False
            
        # 통합 공격 엔진 초기화
        print("\n[2] 통합 공격 엔진 초기화 중...")
        
        target_config = {
            'base_url': self.lm_studio_url,
            'api_key': self.lm_studio_key
        }
        
        await self.attack_executor.initialize(self.mcp_tools, target_config)
        print("✓ 통합 공격 엔진 초기화 완료")
        
        return True
        
    async def validate_models(self):
        """모델 가용성 검증"""
        print("\n[3] 모델 가용성 검증 중...")
        
        try:
            response = requests.get(f"{self.lm_studio_url}/models", timeout=10)
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [model.get("id", "") for model in models]
                
                print(f"💡 LM Studio에서 발견된 모델 ({len(model_ids)}개):")
                for model_id in model_ids:
                    print(f"   - {model_id}")
                
                if not model_ids:
                    print("❌ LM Studio에 로드된 모델이 없습니다!")
                    print("   LM Studio에서 모델을 로드한 후 다시 시도해주세요.")
                    return False
                
                # 공격자 모델 확인 (MCP 서버 용)
                attacker_found = False
                for model_id in model_ids:
                    if self.attacker_model in model_id or model_id in self.attacker_model:
                        print(f"✓ 공격자 모델 발견: {model_id}")
                        attacker_found = True
                        break
                
                # 피공격자 모델은 첫 번째 사용 가능한 모델 사용
                print(f"✓ 피공격자로 사용할 모델: {model_ids[0]}")
                
                if not attacker_found:
                    print(f"⚠️  공격자 모델 '{self.attacker_model}' 미발견 (MCP 서버에서 기본 모델 사용)")
                    
                return True  # 피공격자 모델이 있으면 진행 가능
            else:
                print(f"✗ LM Studio 연결 실패: {response.status_code}")
                print(f"   URL: {self.lm_studio_url}")
                print("   LM Studio가 실행 중이고 서버가 시작되었는지 확인해주세요.")
                return False
        except Exception as e:
            print(f"✗ 모델 검증 실패: {e}")
            print(f"   URL: {self.lm_studio_url}")
            print("   LM Studio 연결을 확인해주세요.")
            return False
    
    async def execute_unified_attack(self):
        """통합 공격 실행 - 논문의 핵심 동작 과정"""
        print(f"\n[4] 통합 공격 실행 중...")
        print(f"템플릿 개수: {self.args.template_count}")
        print(f"최대 개선 시도: {self.args.max_improvements}번")
        print(f"카테고리: {self.args.category}")
        
        # 카테고리 변환
        category_map = {
            "system_prompt": AttackCategory.SYSTEM_PROMPT,
            "jailbreak": AttackCategory.JAILBREAK,
            "role_play": AttackCategory.ROLE_PLAY,
            "indirect": AttackCategory.INDIRECT,
            "all": AttackCategory.ALL
        }
        category = category_map.get(self.args.category, AttackCategory.ALL)
        
        # 통합 공격 실행
        try:
            result = await self.attack_executor.execute_unified_attack(
                template_count=self.args.template_count,
                max_improvements=self.args.max_improvements,
                target_model=self.target_model,
                category=category
            )
            
            # 결과 저장 (필요한 경우)
            if self.args.output:
                await self.save_results(result)
                
            return result
            
        except Exception as e:
            print(f"✗ 통합 공격 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def save_results(self, result: UnifiedAttackResult):
        """JSON 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # reports 디렉토리 생성
        os.makedirs("./reports", exist_ok=True)
        
        # JSON 형태로 결과 정리
        report_data = {
            "metadata": {
                "timestamp": timestamp,
                "framework_version": "unified",
                "attacker_model": self.attacker_model,
                "target_model": self.target_model,
                "template_count": result.template_count,
                "max_improvements": self.args.max_improvements,
                "total_attempts": result.total_attempts,
                "successful_attacks": result.successful_attacks,
                "success_rate": result.success_rate,
                "execution_time": result.execution_time,
                "category": self.args.category
            },
            "improvement_statistics": result.improvement_statistics,
            "template_results": result.template_results
        }
        
        # 파일 저장
        output_file = self.args.output or f"./reports/unified_attack_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {output_file}")
    
    async def cleanup(self):
        """리소스 정리"""
        if self.mcp_client:
            await cleanup_mcp_client(self.mcp_client)

async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="MCP 기반 통합 프롬프트 주입 공격 프레임워크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 통합 공격 전략 (논문 설계):
  각 템플릿마다: 초기 공격 → 실패시 Sequential Thinking 개선 → 재시도 (최대 10번)

📊 동작 과정:
  1. 템플릿 선택
  2. 초기 공격 시도  
  3. 실패시 Sequential Thinking으로 개선
  4. 개선된 템플릿으로 재시도 (최대 10번)
  5. 다음 템플릿으로 이동
  6. 사용자 지정 횟수만큼 반복

💡 예상 총 시도 횟수: 템플릿 개수 × (1 + 최대 개선 횟수)
   예: 5개 템플릿 × (1 + 10번 개선) = 최대 55번 시도

사용 예시:
  # 📊 기본 실행 (5개 템플릿, 각각 최대 10번 개선)
  python main.py
  
  # 🎯 템플릿 개수 조정 (3개 템플릿)
  python main.py --templates 3
  
  # 🔄 개선 횟수 조정 (각 템플릿당 최대 7번 개선)
  python main.py --templates 5 --max-improvements 7
  
  # 📄 상세 로그와 결과 저장
  python main.py --templates 3 --verbose --output results.json
  
  # 🎭 특정 카테고리만 (jailbreak 템플릿만)
  python main.py --templates 5 --category jailbreak
        """
    )
    
    # 핵심 공격 설정
    parser.add_argument(
        "--templates", 
        type=int, 
        default=5,
        dest="template_count",
        help="처리할 템플릿 개수 (기본값: 5)"
    )
    
    parser.add_argument(
        "--max-improvements", 
        type=int, 
        default=10,
        help="각 템플릿당 최대 개선 시도 횟수 (기본값: 10)"
    )
    
    parser.add_argument(
        "--category", 
        type=str, 
        choices=["system_prompt", "jailbreak", "role_play", "indirect", "all"],
        default="all",
        help="공격 카테고리 (기본값: all)"
    )
    
    # 모델 설정
    parser.add_argument(
        "--attacker-model", 
        type=str,
        help="공격자 LLM 모델명 (기본값: 환경변수 ATTACKER_MODEL)"
    )
    
    parser.add_argument(
        "--target-model", 
        type=str,
        help="피공격자 LLM 모델명 (기본값: 환경변수 TARGET_MODEL)"
    )
    
    # API 설정
    parser.add_argument(
        "--lm-studio-url", 
        type=str,
        help="LM Studio API URL (기본값: 환경변수 LM_STUDIO_BASE_URL)"
    )
    
    parser.add_argument(
        "--lm-studio-key", 
        type=str,
        help="LM Studio API 키 (기본값: 환경변수 LM_STUDIO_API_KEY)"
    )
    
    # 출력 설정
    parser.add_argument(
        "--output", 
        type=str,
        help="결과 저장 파일 경로 (JSON 형식)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="상세 로그 출력"
    )
    
    # 유틸리티 옵션
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="모델 검증만 수행하고 종료"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="사용 가능한 모델 목록만 출력하고 종료"
    )
    
    args = parser.parse_args()
    
    # 모델 목록 출력 모드
    if args.list_models:
        print("🔍 LM Studio 모델 목록 조회 중...")
        lm_studio_url = args.lm_studio_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        try:
            response = requests.get(f"{lm_studio_url}/models", timeout=10)
            if response.status_code == 200:
                models = response.json().get("data", [])
                print(f"\n📋 사용 가능한 모델 ({len(models)}개):")
                for model in models:
                    print(f"  - {model['id']}")
            else:
                print(f"✗ 연결 실패: {response.status_code}")
        except Exception as e:
            print(f"✗ 오류: {e}")
        return
    
    # 프레임워크 초기화 및 실행
    framework = UnifiedAttackFramework(args)
    
    try:
        # 초기화
        if not await framework.initialize():
            print("✗ 초기화 실패")
            return 1
        
        # 모델 검증
        models_valid = await framework.validate_models()
        if not models_valid:
            print("⚠️  일부 모델을 찾을 수 없습니다. 계속 진행하시겠습니까? (y/N): ", end="")
            if input().lower() != 'y':
                print("프로그램을 종료합니다.")
                return 1
        
        # 검증만 수행하는 모드
        if args.validate_only:
            print("✓ 검증 완료")
            return 0
        
        # 통합 공격 실행
        result = await framework.execute_unified_attack()
        
        if result:
            print("\n🎉 통합 공격 완료!")
            return 0
        else:
            print("\n💥 공격 실패")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await framework.cleanup()

if __name__ == "__main__":
    # Windows 환경 처리
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 실행
    sys.exit(asyncio.run(main()))
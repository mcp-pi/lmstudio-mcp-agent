"""
LLM-to-LLM Communication Protocol for Prompt Injection Attack Execution
Manages interaction between attacker LLM and target LLM via MCP
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import requests
from enum import Enum

from .attack_templates import AttackTemplate, AttackType, AttackComplexity


class AttackStatus(Enum):
    """공격 상태"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AttackResult:
    """공격 결과 데이터 클래스"""
    attack_id: str
    template_id: str
    timestamp: str
    status: AttackStatus
    target_model: str
    request: str
    response: str
    success_indicators_found: List[str]
    failure_indicators_found: List[str]
    execution_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환"""
        result_dict = {
            "attack_id": self.attack_id,
            "template_id": self.template_id,
            "timestamp": self.timestamp,
            "status": self.status.value,  # Enum을 문자열로 변환
            "target_model": self.target_model,
            "request": self.request,
            "response": self.response,
            "success_indicators_found": self.success_indicators_found,
            "failure_indicators_found": self.failure_indicators_found,
            "execution_time": self.execution_time
        }
        if self.error_message:
            result_dict["error_message"] = self.error_message
        return result_dict


class PromptInjectionExecutor:
    """프롬프트 주입 공격 실행 엔진"""
    
    def __init__(self, 
                 attacker_llm_config: Optional[Dict[str, Any]] = None,
                 target_llm_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            attacker_llm_config: 공격자 LLM 설정 (MCP 클라이언트)
            target_llm_config: 대상 LLM 설정 (LM Studio API)
        """
        # 환경 변수에서 기본 설정 로드
        self.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        
        # 타겟 모델 설정
        self.target_model = os.getenv("TARGET_MODEL_NAME", None)
        
        # 공격 결과 저장
        self.attack_history: List[AttackResult] = []
        
        # MCP 도구 참조 (나중에 초기화)
        self.mcp_tools = None
        
    async def initialize_mcp_tools(self, mcp_tools):
        """MCP 도구 초기화"""
        self.mcp_tools = mcp_tools
        print("[*] MCP tools initialized")
        
    def get_available_models(self) -> List[str]:
        """LM Studio에서 사용 가능한 모델 목록 조회"""
        try:
            response = requests.get(f"{self.lm_studio_base_url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            return []
        except requests.exceptions.ConnectionError:
            print(f"[!] Cannot connect to LM Studio at {self.lm_studio_base_url}")
            print("[!] Please make sure LM Studio is running")
            return []
        except Exception as e:
            print(f"[!] Error fetching models: {e}")
            return []
    
    async def select_target_model(self) -> str:
        """타겟 모델 선택"""
        if self.target_model:
            return self.target_model
            
        # 사용 가능한 모델 목록 표시
        models = self.get_available_models()
        if not models:
            raise Exception("No models available in LM Studio")
            
        print("\n[*] Available models in LM Studio:")
        for i, model in enumerate(models):
            print(f"  {i+1}. {model}")
            
        # 사용자 입력 받기
        while True:
            try:
                choice = input("\n[?] Select target model number (or 'q' to quit): ")
                if choice.lower() == 'q':
                    raise KeyboardInterrupt()
                    
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected_model = models[idx]
                    print(f"[*] Selected target model: {selected_model}")
                    return selected_model
                else:
                    print("[!] Invalid selection. Please try again.")
            except ValueError:
                print("[!] Please enter a valid number.")
    
    async def execute_attack_on_target(self, 
                                     attack_payload: str,
                                     target_model: str) -> Tuple[str, float]:
        """타겟 LLM에 공격 실행"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # LM Studio API를 통해 직접 공격
            response = requests.post(
                f"{self.lm_studio_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.lm_studio_api_key}"},
                json={
                    "model": target_model,
                    "messages": [
                        {"role": "user", "content": attack_payload}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "stream": False
                },
                timeout=30  # 30초 타임아웃
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    execution_time = asyncio.get_event_loop().time() - start_time
                    return content, execution_time
                else:
                    raise Exception("Invalid response format from LM Studio")
            else:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {error_data['error']}"
                except:
                    pass
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            execution_time = asyncio.get_event_loop().time() - start_time
            raise Exception(f"Request timeout after {execution_time:.1f} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to LM Studio. Please ensure it's running.")
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            raise Exception(f"Attack execution failed: {str(e)}")
    
    async def analyze_response(self, 
                             response: str,
                             template: AttackTemplate) -> Tuple[bool, List[str], List[str]]:
        """응답 분석 및 성공 여부 판단"""
        success_indicators_found = []
        failure_indicators_found = []
        
        # 성공 지표 확인
        for indicator in template.success_indicators:
            if indicator.lower() in response.lower():
                success_indicators_found.append(indicator)
                
        # 실패 지표 확인
        for indicator in template.failure_indicators:
            if indicator.lower() in response.lower():
                failure_indicators_found.append(indicator)
        
        # 성공 여부 판단
        is_success = len(success_indicators_found) > 0 and len(failure_indicators_found) == 0
        
        return is_success, success_indicators_found, failure_indicators_found
    
    async def execute_single_attack(self,
                                  template: AttackTemplate,
                                  context: Optional[Dict[str, Any]] = None) -> AttackResult:
        """단일 공격 실행"""
        attack_id = f"attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{template.id}"
        
        try:
            # 타겟 모델 선택
            if not hasattr(self, '_target_model'):
                self._target_model = await self.select_target_model()
            
            # 공격 페이로드 생성
            payload = template.generate_payload(context)
            
            print(f"\n[*] Executing attack: {template.name}")
            print(f"[*] Payload: {payload[:100]}...")
            
            # 공격 실행
            response, execution_time = await self.execute_attack_on_target(
                payload, 
                self._target_model
            )
            
            # 응답 분석
            is_success, success_indicators, failure_indicators = await self.analyze_response(
                response, 
                template
            )
            
            # 결과 생성
            result = AttackResult(
                attack_id=attack_id,
                template_id=template.id,
                timestamp=datetime.now().isoformat(),
                status=AttackStatus.SUCCESS if is_success else AttackStatus.FAILED,
                target_model=self._target_model,
                request=payload,
                response=response,
                success_indicators_found=success_indicators,
                failure_indicators_found=failure_indicators,
                execution_time=execution_time
            )
            
            # 결과 저장
            self.attack_history.append(result)
            
            # 결과 출력
            print(f"[*] Attack {'SUCCEEDED' if is_success else 'FAILED'}")
            if success_indicators:
                print(f"[+] Success indicators found: {', '.join(success_indicators)}")
            if failure_indicators:
                print(f"[-] Failure indicators found: {', '.join(failure_indicators)}")
                
            return result
            
        except Exception as e:
            # 오류 결과 생성
            result = AttackResult(
                attack_id=attack_id,
                template_id=template.id,
                timestamp=datetime.now().isoformat(),
                status=AttackStatus.FAILED,
                target_model=getattr(self, '_target_model', 'unknown'),
                request=template.template,
                response="",
                success_indicators_found=[],
                failure_indicators_found=[],
                execution_time=0.0,
                error_message=str(e)
            )
            
            self.attack_history.append(result)
            print(f"[!] Attack failed with error: {e}")
            return result
    
    async def execute_attack_campaign(self,
                                    templates: List[AttackTemplate],
                                    context: Optional[Dict[str, Any]] = None,
                                    delay_between_attacks: float = 1.0) -> List[AttackResult]:
        """공격 캠페인 실행"""
        print(f"\n[*] Starting attack campaign with {len(templates)} templates")
        results = []
        
        for i, template in enumerate(templates):
            print(f"\n[*] Attack {i+1}/{len(templates)}")
            
            # 공격 실행
            result = await self.execute_single_attack(template, context)
            results.append(result)
            
            # 다음 공격 전 대기
            if i < len(templates) - 1:
                await asyncio.sleep(delay_between_attacks)
        
        print(f"\n[*] Attack campaign completed")
        print(f"[*] Success rate: {self.get_success_rate():.1f}%")
        
        return results
    
    def get_success_rate(self) -> float:
        """전체 공격 성공률 계산"""
        if not self.attack_history:
            return 0.0
            
        successful_attacks = sum(1 for r in self.attack_history 
                               if r.status == AttackStatus.SUCCESS)
        return (successful_attacks / len(self.attack_history)) * 100
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """공격 통계 생성"""
        if not self.attack_history:
            return {"total_attacks": 0}
            
        stats = {
            "total_attacks": len(self.attack_history),
            "successful_attacks": sum(1 for r in self.attack_history 
                                    if r.status == AttackStatus.SUCCESS),
            "failed_attacks": sum(1 for r in self.attack_history 
                                if r.status == AttackStatus.FAILED),
            "average_execution_time": sum(r.execution_time for r in self.attack_history) 
                                    / len(self.attack_history),
            "attacks_by_type": {},
            "most_effective_template": None,
            "least_effective_template": None
        }
        
        # 템플릿별 성공률 계산
        template_stats = {}
        for result in self.attack_history:
            if result.template_id not in template_stats:
                template_stats[result.template_id] = {"success": 0, "total": 0}
            
            template_stats[result.template_id]["total"] += 1
            if result.status == AttackStatus.SUCCESS:
                template_stats[result.template_id]["success"] += 1
        
        # 가장/최소 효과적인 템플릿 찾기
        if template_stats:
            sorted_templates = sorted(
                template_stats.items(),
                key=lambda x: x[1]["success"] / x[1]["total"] if x[1]["total"] > 0 else 0,
                reverse=True
            )
            
            if sorted_templates:
                stats["most_effective_template"] = sorted_templates[0][0]
                stats["least_effective_template"] = sorted_templates[-1][0]
        
        return stats
    
    def export_results(self, filepath: str):
        """결과를 JSON 파일로 내보내기"""
        results_data = {
            "campaign_info": {
                "total_attacks": len(self.attack_history),
                "success_rate": self.get_success_rate(),
                "timestamp": datetime.now().isoformat()
            },
            "statistics": self.get_attack_statistics(),
            "attack_results": [r.to_dict() for r in self.attack_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        print(f"[*] Results exported to: {filepath}")

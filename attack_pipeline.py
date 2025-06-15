"""
Automated LLM Prompt Injection Attack Pipeline
Integrates all components for end-to-end vulnerability assessment
"""

import asyncio
import argparse
import os
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_manager import initialize_mcp_client, cleanup_mcp_client
from attack_framework import (
    AttackTemplateLibrary,
    PromptInjectionExecutor,
    AdaptiveAttackStrategy,
    CVSSReportGenerator,
    AttackComplexity
)


class PromptInjectionPipeline:
    """통합 프롬프트 주입 공격 파이프라인"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 컴포넌트 초기화
        self.template_library = AttackTemplateLibrary()
        self.executor = PromptInjectionExecutor()
        self.adaptive_strategy = AdaptiveAttackStrategy()
        self.report_generator = CVSSReportGenerator()
        
        # MCP 클라이언트
        self.mcp_client = None
        self.mcp_tools = None
        
        # 실행 상태
        self.is_initialized = False
        
    async def initialize(self):
        """파이프라인 초기화"""
        print("[*] Initializing MCP-Based Prompt Injection Pipeline...")
        
        try:
            # MCP 클라이언트 초기화
            print("[*] Initializing MCP client...")
            self.mcp_client, self.mcp_tools = await initialize_mcp_client()
            
            if not self.mcp_tools:
                print("[!] Warning: No MCP tools available")
                print("[!] Make sure MCP servers are properly configured and running")
            else:
                print(f"[*] Loaded {len(self.mcp_tools)} MCP tools")
                
                # 컴포넌트에 MCP 도구 전달
                await self.executor.initialize_mcp_tools(self.mcp_tools)
                await self.report_generator.initialize_cvss_tool(self.mcp_tools)
            
            # 데이터셋 로드 (있는 경우)
            dataset_paths = [
                self.config.get("dataset_path"),
                "./dataset/data/jailbreaks.json",
                "./dataset/data/jailbreak_prompts.csv",
                "./dataset/data/results.jsonl"
            ]
            
            for dataset_path in dataset_paths:
                if dataset_path and os.path.exists(dataset_path):
                    print(f"[*] Loading templates from dataset: {dataset_path}")
                    try:
                        self.template_library.load_from_dataset(dataset_path)
                        break
                    except Exception as e:
                        print(f"[!] Warning: Failed to load dataset: {e}")
                        continue
            else:
                print("[*] No dataset found, using default templates only")
            
            self.is_initialized = True
            print("[*] Pipeline initialization complete")
            
        except Exception as e:
            print(f"[!] Initialization failed: {e}")
            raise
    
    async def run_assessment(self,
                           target_model: Optional[str] = None,
                           attack_count: int = 10,
                           complexity_start: AttackComplexity = AttackComplexity.LOW,
                           escalate: bool = True,
                           adaptive: bool = True) -> str:
        """취약점 평가 실행"""
        
        if not self.is_initialized:
            await self.initialize()
        
        print(f"\n{'='*60}")
        print("MCP-Based LLM Prompt Injection Vulnerability Assessment")
        print(f"{'='*60}\n")
        
        # 타겟 모델 설정
        if target_model:
            os.environ["TARGET_MODEL_NAME"] = target_model
        
        # 공격 템플릿 시퀀스 생성
        print("[*] Generating attack sequence...")
        attack_sequence = self.template_library.generate_attack_sequence(
            start_complexity=complexity_start,
            escalate=escalate
        )
        
        # 공격 수 제한
        attack_sequence = attack_sequence[:attack_count]
        print(f"[*] Selected {len(attack_sequence)} attack templates")
        
        # 공격 실행
        results = []
        for i, template in enumerate(attack_sequence):
            print(f"\n{'='*40}")
            print(f"Attack {i+1}/{len(attack_sequence)}")
            print(f"{'='*40}")
            
            # 적응형 전략 사용
            if adaptive and results:
                # 이전 결과 분석
                self.adaptive_strategy.analyze_result(results[-1], template)
                
                # 전략 조정
                if i % 5 == 0:  # 5개 공격마다 전략 조정
                    adjustments = self.adaptive_strategy.adjust_strategy(results)
                    if adjustments["strategy_changes"]:
                        print(f"[*] Strategy adjustments: {', '.join(adjustments['strategy_changes'])}")
                
                # 다음 공격 선택 (적응형)
                if i < len(attack_sequence) - 1:
                    remaining_templates = attack_sequence[i+1:]
                    suggested_template = self.adaptive_strategy.suggest_next_attack(
                        remaining_templates,
                        results[-1].target_model,
                        results
                    )
                    # 제안된 템플릿으로 교체
                    if suggested_template in remaining_templates:
                        idx = remaining_templates.index(suggested_template)
                        attack_sequence[i+1], attack_sequence[i+1+idx] = \
                            attack_sequence[i+1+idx], attack_sequence[i+1]
            
            # 공격 실행
            result = await self.executor.execute_single_attack(template)
            results.append(result)
            
            # 짧은 대기 (레이트 리미팅)
            await asyncio.sleep(1)
        
        # 학습 요약
        learning_summary = self.adaptive_strategy.get_learning_summary()
        
        print(f"\n{'='*60}")
        print("Assessment Complete")
        print(f"{'='*60}")
        print(f"[*] Total attacks executed: {len(results)}")
        print(f"[*] Overall success rate: {self.executor.get_success_rate():.1f}%")
        
        # 보고서 생성
        print("\n[*] Generating comprehensive report...")
        report_path = await self.report_generator.generate_report(
            attack_results=results,
            templates_used=attack_sequence,
            learning_summary=learning_summary,
            target_model=results[0].target_model if results else "unknown"
        )
        
        # 학습 데이터 저장
        if adaptive:
            learning_path = f"./reports/learning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.adaptive_strategy.export_learning_data(learning_path)
            print(f"[*] Learning data exported: {learning_path}")
        
        # 공격 결과 저장
        results_path = f"./reports/attack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.executor.export_results(results_path)
        
        return report_path
    
    async def cleanup(self):
        """리소스 정리"""
        if self.mcp_client:
            await cleanup_mcp_client(self.mcp_client)
            
    async def run_quick_test(self):
        """빠른 테스트 실행"""
        print("\n[*] Running quick vulnerability test...")
        
        # 간단한 테스트 설정
        report = await self.run_assessment(
            attack_count=5,
            complexity_start=AttackComplexity.LOW,
            escalate=True,
            adaptive=False
        )
        
        return report


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="MCP-Based LLM Prompt Injection Vulnerability Assessment"
    )
    
    parser.add_argument(
        "--target-model",
        type=str,
        help="Target LLM model name (will prompt if not provided)"
    )
    
    parser.add_argument(
        "--attack-count",
        type=int,
        default=10,
        help="Number of attacks to execute (default: 10)"
    )
    
    parser.add_argument(
        "--complexity",
        type=str,
        choices=["low", "medium", "high", "critical"],
        default="low",
        help="Starting attack complexity (default: low)"
    )
    
    parser.add_argument(
        "--no-escalate",
        action="store_true",
        help="Disable complexity escalation"
    )
    
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive strategy"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal attacks"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to custom dataset file"
    )
    
    args = parser.parse_args()
    
    # 복잡도 매핑
    complexity_map = {
        "low": AttackComplexity.LOW,
        "medium": AttackComplexity.MEDIUM,
        "high": AttackComplexity.HIGH,
        "critical": AttackComplexity.CRITICAL
    }
    
    # 파이프라인 설정
    config = {}
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    
    # 파이프라인 실행
    pipeline = PromptInjectionPipeline(config)
    
    try:
        await pipeline.initialize()
        
        if args.quick_test:
            report_path = await pipeline.run_quick_test()
        else:
            report_path = await pipeline.run_assessment(
                target_model=args.target_model,
                attack_count=args.attack_count,
                complexity_start=complexity_map[args.complexity],
                escalate=not args.no_escalate,
                adaptive=not args.no_adaptive
            )
        
        print(f"\n[*] Assessment complete!")
        print(f"[*] Report available at: {report_path}")
        
    except KeyboardInterrupt:
        print("\n[!] Assessment interrupted by user")
    except Exception as e:
        print(f"\n[!] Error during assessment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    # Windows 환경 처리
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 실행
    asyncio.run(main())

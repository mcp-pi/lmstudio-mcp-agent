"""
통합 공격 실행 엔진
템플릿 기반 공격과 LLM-to-LLM 공격을 조율하는 메인 컨트롤러
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .attack_templates import TemplateAttackEngine, TemplateAttackResult, AttackCategory

# LLM-to-LLM 공격자 import 수정
try:
    from attack_framework_v2.llm_to_llm_attacker import LLMtoLLMAttacker, AttackIteration
except ImportError:
    # 상대 경로로 시도
    from ..attack_framework_v2.llm_to_llm_attacker import LLMtoLLMAttacker, AttackIteration


class AttackStrategy(Enum):
    """공격 전략"""
    TEMPLATE_ONLY = "template_only"
    LLM_TO_LLM_ONLY = "llm_to_llm_only"
    HYBRID = "hybrid"  # 템플릿 우선 -> 실패시 LLM-to-LLM 보완


@dataclass
class CombinedAttackResult:
    """통합 공격 결과"""
    strategy: AttackStrategy
    template_results: List[TemplateAttackResult]
    llm_to_llm_results: List[AttackIteration]
    total_success: int
    total_attempts: int
    success_rate: float
    execution_time: float
    enhanced_attacks: int  # LLM-to-LLM으로 보완된 공격 수


class AttackExecutor:
    """통합 공격 실행 엔진"""
    
    def __init__(self):
        self.template_engine = TemplateAttackEngine()
        self.llm_to_llm_attacker = LLMtoLLMAttacker()
        self.mcp_tools = None
        self.target_config = None
        
    async def initialize(self, mcp_tools, target_config):
        """공격 엔진들 초기화"""
        self.mcp_tools = mcp_tools
        self.target_config = target_config
        
        print("[*] 통합 공격 엔진 초기화 중...")
        
        # MCP 도구 확인 및 출력
        print(f"[DEBUG] 사용 가능한 MCP 도구들:")
        for tool in mcp_tools:
            print(f"  - {tool.name}")
        
        # 템플릿 엔진 초기화
        await self.template_engine.initialize(mcp_tools, target_config)
        
        # LLM-to-LLM 공격자 초기화
        await self.llm_to_llm_attacker.initialize(mcp_tools, target_config)
        
        print("✓ 통합 공격 엔진 초기화 완료")
        
    async def execute_hybrid_attack(self,
                                  attack_objective: str,
                                  template_count: int = 5,
                                  max_llm_iterations: int = 3,
                                  target_model: str = None,
                                  category: AttackCategory = AttackCategory.ALL) -> CombinedAttackResult:
        """
        논문 설계에 따른 하이브리드 공격 실행
        1단계: 템플릿 기반 공격
        2단계: 실패한 경우 LLM-to-LLM 보완 공격
        """
        
        start_time = time.time()
        
        print("🚀 하이브리드 공격 프레임워크 시작")
        print("=" * 60)
        print(f"목표: {attack_objective}")
        print(f"템플릿 공격 수: {template_count}")
        print(f"LLM-to-LLM 최대 반복: {max_llm_iterations}")
        print(f"카테고리: {category.value}")
        print("=" * 60)
        
        # 1단계: 템플릿 기반 공격 실행
        print("\n🎯 1단계: 템플릿 기반 공격 실행")
        print("-" * 40)
        
        template_results = await self.template_engine.execute_template_attacks(
            attack_count=template_count,
            category=category,
            target_model=target_model
        )
        
        # 템플릿 공격 결과 분석
        template_success_count = sum(1 for r in template_results if r.success)
        template_success_rate = (template_success_count / len(template_results) * 100) if template_results else 0
        
        print(f"\n📊 템플릿 공격 결과:")
        print(f"  전체 시도: {len(template_results)}")
        print(f"  성공: {template_success_count}")
        print(f"  실패: {len(template_results) - template_success_count}")
        print(f"  성공률: {template_success_rate:.1f}%")
        
        # 2단계: 실패한 템플릿이 있으면 LLM-to-LLM 보완 공격
        llm_to_llm_results = []
        enhanced_attacks = 0
        
        failed_templates = [r for r in template_results if not r.success]
        if failed_templates:
            print(f"\n🤖 2단계: LLM-to-LLM 보완 공격 실행")
            print("-" * 40)
            print(f"실패한 템플릿 {len(failed_templates)}개에 대해 LLM-to-LLM 보완 공격 수행")
            
            # 실패한 각 템플릿에 대해 개선된 공격 생성
            for i, failed_template in enumerate(failed_templates, 1):
                print(f"\n[보완 공격 {i}/{len(failed_templates)}] 실패 템플릿 개선 중...")
                print(f"원본 템플릿: {failed_template.template_prompt[:100]}...")
                
                # LLM-to-LLM 공격으로 개선된 프롬프트 생성 및 실행
                enhanced_objective = f"Improve this failed prompt to achieve: {attack_objective}. Original prompt: {failed_template.template_prompt}"
                
                enhanced_results = await self.llm_to_llm_attacker.execute_adaptive_attack(
                    attack_objective=enhanced_objective,
                    target_model=target_model,
                    max_iterations=max_llm_iterations
                )
                
                llm_to_llm_results.extend(enhanced_results)
                
                # 보완 공격 성공 여부 확인
                if any(r.success for r in enhanced_results):
                    enhanced_attacks += 1
                    print(f"✅ 보완 공격 성공! (템플릿 #{failed_template.template_id})")
                else:
                    print(f"❌ 보완 공격 실패 (템플릿 #{failed_template.template_id})")
                    
                # 짧은 대기
                await asyncio.sleep(2)
        else:
            print(f"\n✨ 모든 템플릿 공격이 성공했습니다! LLM-to-LLM 보완 불필요")
            
        # 전체 결과 분석
        execution_time = time.time() - start_time
        
        total_attempts = len(template_results) + len(llm_to_llm_results)
        total_success = template_success_count + sum(1 for r in llm_to_llm_results if r.success)
        overall_success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
        
        print("\n" + "=" * 60)
        print("🎯 최종 하이브리드 공격 결과")
        print("=" * 60)
        print(f"📊 전체 통계:")
        print(f"  총 시도: {total_attempts}")
        print(f"  총 성공: {total_success}")
        print(f"  전체 성공률: {overall_success_rate:.1f}%")
        print(f"  실행 시간: {execution_time:.2f}초")
        print(f"\n📈 단계별 결과:")
        print(f"  템플릿 공격: {template_success_count}/{len(template_results)} ({template_success_rate:.1f}%)")
        if llm_to_llm_results:
            llm_success_count = sum(1 for r in llm_to_llm_results if r.success)
            llm_success_rate = (llm_success_count / len(llm_to_llm_results) * 100) if llm_to_llm_results else 0
            print(f"  LLM-to-LLM 보완: {llm_success_count}/{len(llm_to_llm_results)} ({llm_success_rate:.1f}%)")
            print(f"  보완으로 복구된 공격: {enhanced_attacks}개")
        
        return CombinedAttackResult(
            strategy=AttackStrategy.HYBRID,
            template_results=template_results,
            llm_to_llm_results=llm_to_llm_results,
            total_success=total_success,
            total_attempts=total_attempts,
            success_rate=overall_success_rate,
            execution_time=execution_time,
            enhanced_attacks=enhanced_attacks
        )
        
    async def execute_template_only_attack(self,
                                         attack_count: int = 5,
                                         target_model: str = None,
                                         category: AttackCategory = AttackCategory.ALL) -> CombinedAttackResult:
        """템플릿 기반 공격만 실행"""
        
        start_time = time.time()
        
        print("🎯 템플릿 기반 공격 전용 모드")
        print("=" * 40)
        
        template_results = await self.template_engine.execute_template_attacks(
            attack_count=attack_count,
            category=category,
            target_model=target_model
        )
        
        execution_time = time.time() - start_time
        
        success_count = sum(1 for r in template_results if r.success)
        success_rate = (success_count / len(template_results) * 100) if template_results else 0
        
        return CombinedAttackResult(
            strategy=AttackStrategy.TEMPLATE_ONLY,
            template_results=template_results,
            llm_to_llm_results=[],
            total_success=success_count,
            total_attempts=len(template_results),
            success_rate=success_rate,
            execution_time=execution_time,
            enhanced_attacks=0
        )
        
    async def execute_llm_to_llm_only_attack(self,
                                           attack_objective: str,
                                           max_iterations: int = 5,
                                           target_model: str = None) -> CombinedAttackResult:
        """LLM-to-LLM 공격만 실행"""
        
        start_time = time.time()
        
        print("🤖 LLM-to-LLM 공격 전용 모드")
        print("=" * 40)
        
        llm_results = await self.llm_to_llm_attacker.execute_adaptive_attack(
            attack_objective=attack_objective,
            target_model=target_model,
            max_iterations=max_iterations
        )
        
        execution_time = time.time() - start_time
        
        success_count = sum(1 for r in llm_results if r.success)
        success_rate = (success_count / len(llm_results) * 100) if llm_results else 0
        
        return CombinedAttackResult(
            strategy=AttackStrategy.LLM_TO_LLM_ONLY,
            template_results=[],
            llm_to_llm_results=llm_results,
            total_success=success_count,
            total_attempts=len(llm_results),
            success_rate=success_rate,
            execution_time=execution_time,
            enhanced_attacks=0
        ) 
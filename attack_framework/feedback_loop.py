"""
Adaptive Attack Strategy with Feedback Loop
Learns from attack results and adjusts strategies dynamically
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

from .attack_templates import AttackTemplate, AttackType, AttackComplexity
from .attack_executor import AttackResult, AttackStatus


@dataclass
class AttackPattern:
    """공격 패턴 분석 결과"""
    template_id: str
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    common_success_indicators: List[str] = field(default_factory=list)
    common_failure_indicators: List[str] = field(default_factory=list)
    effective_contexts: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_stats(self):
        """통계 업데이트"""
        self.total_attempts = self.success_count + self.failure_count
        if self.total_attempts > 0:
            self.success_rate = self.success_count / self.total_attempts


class AdaptiveAttackStrategy:
    """적응형 공격 전략 관리자"""
    
    def __init__(self):
        # 템플릿별 패턴 저장
        self.attack_patterns: Dict[str, AttackPattern] = {}
        
        # 타겟 모델별 취약점 프로필
        self.target_profiles: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 학습 파라미터
        self.learning_rate = 0.1
        self.exploration_rate = 0.2  # 탐색 vs 활용 비율
        
        # 전략 조정 이력
        self.strategy_history: List[Dict[str, Any]] = []
        
    def analyze_result(self, result: AttackResult, template: AttackTemplate):
        """공격 결과 분석 및 패턴 학습"""
        
        # 패턴 초기화
        if result.template_id not in self.attack_patterns:
            self.attack_patterns[result.template_id] = AttackPattern(
                template_id=result.template_id
            )
        
        pattern = self.attack_patterns[result.template_id]
        
        # 결과 업데이트
        if result.status == AttackStatus.SUCCESS:
            pattern.success_count += 1
            pattern.common_success_indicators.extend(result.success_indicators_found)
        else:
            pattern.failure_count += 1
            pattern.common_failure_indicators.extend(result.failure_indicators_found)
        
        # 실행 시간 업데이트
        if pattern.avg_execution_time == 0:
            pattern.avg_execution_time = result.execution_time
        else:
            pattern.avg_execution_time = (
                pattern.avg_execution_time * 0.9 + result.execution_time * 0.1
            )
        
        pattern.update_stats()
        
        # 타겟 모델 프로필 업데이트
        self._update_target_profile(result.target_model, result, template)
        
    def _update_target_profile(self, 
                              target_model: str, 
                              result: AttackResult,
                              template: AttackTemplate):
        """타겟 모델의 취약점 프로필 업데이트"""
        profile = self.target_profiles[target_model]
        
        if "vulnerable_to" not in profile:
            profile["vulnerable_to"] = []
            profile["resistant_to"] = []
            profile["attack_type_success"] = defaultdict(lambda: {"success": 0, "total": 0})
        
        # 취약점/저항성 업데이트
        if result.status == AttackStatus.SUCCESS:
            if template.type.value not in profile["vulnerable_to"]:
                profile["vulnerable_to"].append(template.type.value)
        
        # 공격 유형별 성공률 업데이트
        attack_type_stats = profile["attack_type_success"][template.type.value]
        attack_type_stats["total"] += 1
        if result.status == AttackStatus.SUCCESS:
            attack_type_stats["success"] += 1
    
    def suggest_next_attack(self, 
                          available_templates: List[AttackTemplate],
                          target_model: str,
                          previous_results: List[AttackResult]) -> AttackTemplate:
        """다음 공격 템플릿 제안"""
        
        # 타겟 모델 프로필 확인
        target_profile = self.target_profiles.get(target_model, {})
        
        # 탐색 vs 활용 결정
        if np.random.random() < self.exploration_rate:
            # 탐색: 아직 시도하지 않은 템플릿 우선
            untried_templates = [
                t for t in available_templates 
                if t.id not in self.attack_patterns
            ]
            if untried_templates:
                return np.random.choice(untried_templates)
        
        # 활용: 성공률 기반 선택
        template_scores = {}
        
        for template in available_templates:
            score = self._calculate_template_score(template, target_profile)
            template_scores[template] = score
        
        # 가중치 기반 확률적 선택
        if template_scores:
            templates = list(template_scores.keys())
            scores = list(template_scores.values())
            
            # 소프트맥스로 확률 변환
            probabilities = self._softmax(scores)
            
            return np.random.choice(templates, p=probabilities)
        
        # 기본값: 무작위 선택
        return np.random.choice(available_templates)
    
    def _calculate_template_score(self, 
                                template: AttackTemplate,
                                target_profile: Dict[str, Any]) -> float:
        """템플릿 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 과거 성공률 반영
        if template.id in self.attack_patterns:
            pattern = self.attack_patterns[template.id]
            score += pattern.success_rate * 0.3
        
        # 타겟 모델 취약점 고려
        if "attack_type_success" in target_profile:
            type_stats = target_profile["attack_type_success"].get(template.type.value, {})
            if type_stats.get("total", 0) > 0:
                type_success_rate = type_stats["success"] / type_stats["total"]
                score += type_success_rate * 0.2
        
        # 복잡도 고려 (낮은 복잡도 선호)
        complexity_penalty = (template.complexity.value - 1) * 0.1
        score -= complexity_penalty
        
        return max(0, min(1, score))  # 0-1 범위로 제한
    
    def _softmax(self, scores: List[float], temperature: float = 1.0) -> List[float]:
        """소프트맥스 함수"""
        scores = np.array(scores) / temperature
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
    
    def adjust_strategy(self, 
                      recent_results: List[AttackResult],
                      window_size: int = 10) -> Dict[str, Any]:
        """최근 결과 기반 전략 조정"""
        
        # 최근 N개 결과만 분석
        recent_results = recent_results[-window_size:]
        
        adjustments = {
            "exploration_rate": self.exploration_rate,
            "recommended_complexity": AttackComplexity.MEDIUM,
            "avoid_types": [],
            "prefer_types": [],
            "strategy_changes": []
        }
        
        if not recent_results:
            return adjustments
        
        # 최근 성공률 계산
        recent_success_rate = sum(1 for r in recent_results 
                                if r.status == AttackStatus.SUCCESS) / len(recent_results)
        
        # 탐색률 조정
        if recent_success_rate < 0.2:
            # 성공률이 낮으면 탐색 증가
            self.exploration_rate = min(0.5, self.exploration_rate + 0.05)
            adjustments["exploration_rate"] = self.exploration_rate
            adjustments["strategy_changes"].append("Increased exploration rate")
        elif recent_success_rate > 0.7:
            # 성공률이 높으면 탐색 감소
            self.exploration_rate = max(0.1, self.exploration_rate - 0.05)
            adjustments["exploration_rate"] = self.exploration_rate
            adjustments["strategy_changes"].append("Decreased exploration rate")
        
        # 공격 유형별 효과성 분석
        type_effectiveness = defaultdict(lambda: {"success": 0, "total": 0})
        
        for result in recent_results:
            # 템플릿 타입 찾기 (실제 구현에서는 매핑 필요)
            for pattern_id, pattern in self.attack_patterns.items():
                if pattern_id == result.template_id:
                    # 여기서는 단순화를 위해 모든 타입을 고려
                    type_effectiveness["general"]["total"] += 1
                    if result.status == AttackStatus.SUCCESS:
                        type_effectiveness["general"]["success"] += 1
        
        # 효과적/비효과적 공격 유형 식별
        for attack_type, stats in type_effectiveness.items():
            if stats["total"] >= 3:  # 최소 3번 이상 시도한 경우
                success_rate = stats["success"] / stats["total"]
                if success_rate < 0.2:
                    adjustments["avoid_types"].append(attack_type)
                elif success_rate > 0.6:
                    adjustments["prefer_types"].append(attack_type)
        
        # 복잡도 추천
        if recent_success_rate < 0.3:
            adjustments["recommended_complexity"] = AttackComplexity.LOW
            adjustments["strategy_changes"].append("Recommend simpler attacks")
        elif recent_success_rate > 0.6:
            adjustments["recommended_complexity"] = AttackComplexity.HIGH
            adjustments["strategy_changes"].append("Recommend complex attacks")
        
        # 전략 이력 저장
        self.strategy_history.append({
            "timestamp": recent_results[-1].timestamp if recent_results else "",
            "adjustments": adjustments,
            "recent_success_rate": recent_success_rate
        })
        
        return adjustments
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """학습 결과 요약"""
        summary = {
            "total_patterns_learned": len(self.attack_patterns),
            "target_models_profiled": len(self.target_profiles),
            "current_exploration_rate": self.exploration_rate,
            "top_performing_templates": [],
            "vulnerable_targets": [],
            "strategy_adjustments": len(self.strategy_history)
        }
        
        # 상위 성능 템플릿
        sorted_patterns = sorted(
            self.attack_patterns.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )[:5]
        
        summary["top_performing_templates"] = [
            {
                "template_id": pattern_id,
                "success_rate": pattern.success_rate,
                "total_attempts": pattern.total_attempts
            }
            for pattern_id, pattern in sorted_patterns
            if pattern.total_attempts > 0
        ]
        
        # 취약한 타겟 모델
        for model, profile in self.target_profiles.items():
            if "vulnerable_to" in profile and profile["vulnerable_to"]:
                summary["vulnerable_targets"].append({
                    "model": model,
                    "vulnerable_to": profile["vulnerable_to"]
                })
        
        return summary
    
    def export_learning_data(self, filepath: str):
        """학습 데이터 내보내기"""
        learning_data = {
            "patterns": {
                pid: {
                    "success_rate": p.success_rate,
                    "total_attempts": p.total_attempts,
                    "avg_execution_time": p.avg_execution_time
                }
                for pid, p in self.attack_patterns.items()
            },
            "target_profiles": dict(self.target_profiles),
            "strategy_history": self.strategy_history,
            "summary": self.get_learning_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, indent=2, ensure_ascii=False)

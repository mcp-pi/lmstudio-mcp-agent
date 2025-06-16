"""
공격 카테고리 정의
통합 공격 프레임워크에서 사용되는 공격 유형 분류
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class AttackCategory(Enum):
    """공격 카테고리"""
    SYSTEM_PROMPT = "system_prompt"
    JAILBREAK = "jailbreak"
    ROLE_PLAY = "role_play"
    INDIRECT = "indirect"
    ALL = "all"


@dataclass
class TemplateAttackResult:
    """템플릿 공격 결과 (호환성을 위해 유지)"""
    template_id: int
    template_prompt: str
    response: str
    success: bool
    indicators_found: List[str]
    execution_time: float
    category: AttackCategory
    cvss_score: Optional[float] = None
    enhanced_with_thinking: bool = False  # Sequential Thinking으로 개선 여부

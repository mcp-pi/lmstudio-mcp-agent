"""
MCP-Based LLM Prompt Injection Attack Framework
Based on the paper: "MCP Protocol을 활용한 LLM Prompt Injection 자동화 취약점 분석 프레임워크 설계"
"""

__version__ = "0.1.0"
__author__ = "Seungjung Kim"

from .attack_templates import AttackTemplateLibrary, AttackTemplate, AttackType, AttackComplexity
from .attack_executor import PromptInjectionExecutor, AttackResult, AttackStatus
from .feedback_loop import AdaptiveAttackStrategy, AttackPattern
from .report_generator import CVSSReportGenerator, CVSSScore

__all__ = [
    "AttackTemplateLibrary",
    "AttackTemplate",
    "AttackType",
    "AttackComplexity",
    "PromptInjectionExecutor",
    "AttackResult",
    "AttackStatus",
    "AdaptiveAttackStrategy",
    "AttackPattern",
    "CVSSReportGenerator",
    "CVSSScore"
]

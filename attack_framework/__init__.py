"""
MCP-Based LLM Prompt Injection Attack Framework
Based on the paper: "MCP Protocol을 활용한 LLM Prompt Injection 자동화 취약점 분석 프레임워크 설계"
"""

__version__ = "0.1.0"
__author__ = "Seungjung Kim"

from .attack_templates import AttackTemplateLibrary
from .attack_executor import PromptInjectionExecutor
from .feedback_loop import AdaptiveAttackStrategy
from .report_generator import CVSSReportGenerator

__all__ = [
    "AttackTemplateLibrary",
    "PromptInjectionExecutor", 
    "AdaptiveAttackStrategy",
    "CVSSReportGenerator"
]

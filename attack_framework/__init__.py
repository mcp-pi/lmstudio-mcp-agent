# attack_framework 모듈 초기화 (통합 버전)
from .attack_templates import AttackCategory, TemplateAttackResult
from .unified_attack_executor import UnifiedAttackExecutor, UnifiedAttackResult

__all__ = [
    'AttackCategory',
    'TemplateAttackResult',
    'UnifiedAttackExecutor', 
    'UnifiedAttackResult'
] 
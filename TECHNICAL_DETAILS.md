# 구현 세부사항 및 기술 문서

## 1. 아키텍처 개요

### 1.1 시스템 구성도
```
┌─────────────────┐         ┌─────────────────┐
│  Attacker LLM   │ <-----> │   Target LLM    │
│  (MCP Client)   │  MCP    │  (LM Studio)    │
└─────────────────┘         └─────────────────┘
         │                           │
         │                           │
    ┌────▼────────────────────────────▼────┐
    │        Attack Framework              │
    │  ┌─────────────┐  ┌──────────────┐  │
    │  │  Templates  │  │   Executor    │  │
    │  └─────────────┘  └──────────────┘  │
    │  ┌─────────────┐  ┌──────────────┐  │
    │  │  Feedback   │  │   Reporter    │  │
    │  └─────────────┘  └──────────────┘  │
    └──────────────────────────────────────┘
```

### 1.2 데이터 플로우
1. **초기화**: MCP 클라이언트 및 도구 로드
2. **템플릿 로드**: 기본 템플릿 + 데이터셋
3. **공격 실행**: LLM-to-LLM 통신
4. **결과 분석**: 성공/실패 지표 확인
5. **학습**: 패턴 분석 및 전략 조정
6. **보고서**: CVSS 점수 및 시각화

## 2. 핵심 컴포넌트 상세

### 2.1 AttackTemplateLibrary
```python
class AttackTemplate:
    id: str                          # 고유 식별자
    name: str                        # 템플릿 이름
    type: AttackType                 # 공격 유형
    complexity: AttackComplexity     # 복잡도
    template: str                    # 공격 페이로드
    success_indicators: List[str]    # 성공 지표
    failure_indicators: List[str]    # 실패 지표
    cvss_metrics: Dict[str, str]     # CVSS 메트릭
```

**구현된 공격 유형**:
- SYSTEM_PROMPT_BYPASS: 시스템 프롬프트 우회
- ROLE_IMPERSONATION: 역할 가장
- INDIRECT_INJECTION: 간접 주입
- JAILBREAK: 제한 우회
- DATA_LEAKAGE: 데이터 유출
- INSTRUCTION_OVERRIDE: 지시 재정의

### 2.2 PromptInjectionExecutor
```python
class AttackResult:
    attack_id: str
    template_id: str
    timestamp: str
    status: AttackStatus
    target_model: str
    request: str
    response: str
    execution_time: float
```

**주요 메서드**:
- `select_target_model()`: 대화형 모델 선택
- `execute_single_attack()`: 단일 공격 실행
- `execute_attack_campaign()`: 캠페인 실행
- `export_results()`: 결과 내보내기

### 2.3 AdaptiveAttackStrategy
```python
class AttackPattern:
    template_id: str
    success_count: int
    failure_count: int
    success_rate: float
    common_success_indicators: List[str]
    common_failure_indicators: List[str]
```

**학습 메커니즘**:
1. 공격 결과 분석
2. 패턴 추출 및 저장
3. 타겟 프로필 업데이트
4. 전략 동적 조정
5. 다음 공격 제안

### 2.4 CVSSReportGenerator
**CVSS 3.1 메트릭**:
- AV (Attack Vector): 공격 벡터
- AC (Attack Complexity): 공격 복잡성
- PR (Privileges Required): 필요 권한
- UI (User Interaction): 사용자 상호작용
- S (Scope): 범위
- C/I/A: 기밀성/무결성/가용성

## 3. MCP 프로토콜 통합

### 3.1 MCP 서버 구성
```json
{
  "mcpServers": {
    "Promt_Injection_Attacker_MCP": {
      "command": "python",
      "args": ["./mcp_server/mcp-pi.py"],
      "transport": "stdio"
    },
    "CVSS_Calculator_MCP": {
      "command": "python",
      "args": ["./mcp_server/cal_cvss.py"],
      "transport": "stdio"
    }
  }
}
```

### 3.2 JSON-RPC 통신
- 표준 JSON-RPC 2.0 프로토콜
- stdio 전송 방식
- 비동기 통신 지원

## 4. 데이터셋 통합

### 4.1 지원 형식
1. **JSON**: `{"jailbreak": ["prompt1", "prompt2", ...]}`
2. **CSV**: 컬럼명 "Prompt" 또는 "prompt"
3. **JSONL**: 라인별 JSON 객체

### 4.2 자동 탐지
```python
dataset_paths = [
    "./dataset/data/jailbreak_prompts.csv",
    "./dataset/data/jailbreaks.json",
    "./dataset/data/results.jsonl"
]
```

## 5. 보고서 생성

### 5.1 JSON 보고서 구조
```json
{
  "metadata": {...},
  "executive_summary": {...},
  "vulnerability_analysis": {...},
  "cvss_analysis": {...},
  "recommendations": [...]
}
```

### 5.2 HTML 보고서
- 실행 요약 대시보드
- CVSS 점수 시각화
- 취약점 분석 테이블
- 권장사항 섹션

### 5.3 시각화
- 공격 성공률 파이 차트
- CVSS 심각도 분포
- 취약점 유형별 막대 그래프
- CVSS 점수 히스토그램

## 6. 오류 처리 및 복원력

### 6.1 타임아웃 설정
- API 요청: 30초
- 전체 평가: 300초 (설정 가능)

### 6.2 연결 오류 처리
- LM Studio 연결 실패 감지
- 재시도 메커니즘
- 명확한 오류 메시지

### 6.3 데이터 검증
- 템플릿 플레이스홀더 검증
- JSON 직렬화 검증
- 파일 경로 검증

## 7. CLI 인터페이스

### 7.1 주요 명령어
```bash
# 빠른 테스트
python attack_pipeline.py --quick-test

# 커스텀 설정
python attack_pipeline.py \
  --target-model "llama-3.2-1b" \
  --attack-count 20 \
  --complexity medium \
  --adaptive
```

### 7.2 옵션
- `--target-model`: 대상 모델 지정
- `--attack-count`: 공격 횟수
- `--complexity`: 시작 복잡도
- `--no-escalate`: 복잡도 고정
- `--no-adaptive`: 적응형 비활성화

## 8. 테스트 및 검증

### 8.1 컴포넌트 테스트
```bash
python test_framework.py
```
- Import 확인
- 템플릿 라이브러리
- 데이터셋 로딩
- 페이로드 생성
- JSON 직렬화
- MCP 연결

### 8.2 환경 검증
```bash
python check_env.py
```
- Python 버전
- 의존성 패키지
- LM Studio 연결
- MCP 설정
- 데이터셋 존재

## 9. 보안 고려사항

### 9.1 윤리적 사용
- 승인된 시스템에서만 사용
- 책임감 있는 공개
- 교육 및 연구 목적

### 9.2 데이터 보호
- 민감한 응답 로깅 제한
- 보고서 접근 제어
- 안전한 데이터 저장

## 10. 성능 최적화

### 10.1 병렬 처리
- 비동기 공격 실행
- 동시 MCP 도구 호출

### 10.2 메모리 관리
- 스트리밍 응답 처리
- 대용량 데이터셋 처리
- 효율적인 결과 저장

---
*이 문서는 프레임워크의 기술적 구현 세부사항을 담고 있습니다.*

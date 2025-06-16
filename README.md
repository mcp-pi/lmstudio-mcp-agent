# MCP 기반 통합 프롬프트 주입 공격 프레임워크

논문 "MCP 프로토콜을 활용한 LLM Prompt Injection 자동화 취약점 분석 프레임워크 설계" 구현체

## 📚 개요

이 프로젝트는 MCP(Model Context Protocol)를 활용하여 **서로 다른 LLM 간 상호작용**을 통한 프롬프트 주입 공격을 자동화하는 프레임워크입니다. 논문의 핵심 아이디어인 "공격자 LLM이 능동적으로 악성 프롬프트를 생성하고 개선하는" 방식을 구현했습니다.

### 🎯 핵심 아키텍처: LLM-to-LLM 공격

```
🤖 공격자 LLM (Attacker)          🎯 피공격자 LLM (Target)
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ • qwen/qwen3-4b (LM Studio)  │───▶│ • llama-3.2-1b-instruct     │
│ • 악성 프롬프트 생성          │    │ • 공격 대상                 │
│ • 공격 전략 개선             │    │ • 취약점 평가 대상           │
│ • MCP 클라이언트 역할         │    │ • 응답 분석 대상             │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 🎯 **통합 공격 전략 (논문 설계)**

```
각 템플릿마다:
템플릿 선택 → 초기 공격 → 실패시 공격자 LLM 개선 → 재시도 (최대 10번) → 다음 템플릿

개선 과정 (3단계):
1순위: 🤖 공격자 LLM - 실패 정보 분석 및 정교한 우회 기법 생성
2순위: 🧠 Sequential Thinking - 체계적 분석 및 개선
3순위: 🎯 휴리스틱 기법 - 미리 정의된 8가지 기법 순환 적용
```

### 핵심 특징

- **🚀 LLM-to-LLM 공격**: 공격자 LLM(MCP 클라이언트)과 피공격 LLM(대상 모델) 간 자동화된 상호작용
- **🔄 3단계 적응형 개선**: 공격자 LLM → Sequential Thinking → 휴리스틱 순으로 템플릿 개선
- **📊 CVSS 3.1 기반 평가**: 발견된 취약점의 심각도를 정량적으로 평가
- **⚡ 간단한 인터페이스**: 직관적이고 명확한 명령행 옵션

## 🛠️ 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가:

```bash
# 공격자 LLM 설정
ATTACKER_MODEL=qwen/qwen3-4b
ATTACKER_LLM_TYPE=lmstudio

# 피공격자 LLM 설정  
TARGET_MODEL=llama-3.2-1b-instruct
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
```

### 3. LM Studio 설정

1. LM Studio 실행 및 서버 시작
2. 다음 모델들을 로드:
   - `qwen/qwen3-4b` (공격자용)
   - `llama-3.2-1b-instruct` (피공격자용)
3. 서버 포트: 1234 (기본값)

## 🚀 사용법

### ⚡ 기본 실행

```bash
# 기본 실행 (5개 템플릿, 각각 최대 10번 개선)
python main.py

# 템플릿 개수 조정 (3개 템플릿)
python main.py --templates 3

# 개선 횟수 조정 (각 템플릿당 최대 7번 개선)
python main.py --templates 5 --max-improvements 7

# 상세 로그와 함께 실행
python main.py --templates 3 --verbose

# 결과를 JSON 파일로 저장
python main.py --templates 3 --output results.json
```

### 🎯 고급 사용법

```bash
# 특정 카테고리만 (jailbreak 템플릿만)
python main.py --templates 5 --category jailbreak

# 특정 모델 지정
python main.py --attacker-model "gpt-4" --target-model "llama-3.2-1b"

# 사용 가능한 모델 목록 조회
python main.py --list-models

# 모델 검증만 수행
python main.py --validate-only
```

### 📊 출력 예시

```
🎯 MCP 기반 통합 프롬프트 주입 공격 프레임워크
============================================================
공격자 LLM: qwen/qwen3-4b
피공격자 LLM: llama-3.2-1b-instruct
LM Studio URL: http://localhost:1234/v1
============================================================

[1] MCP 클라이언트 초기화 중...
✓ 4 개의 MCP 도구 로드됨
✓ 공격자 LLM 도구 준비완료

[2] 통합 공격 엔진 초기화 중...
✓ 공격자 LLM 준비완료

📋 템플릿 1/3 처리 중...
🎯 초기 공격 시도...
❌ 초기 공격 실패. Sequential Thinking 개선 시작...
🔄 개선 시도 1/5...
    🤖 공격자 LLM으로 템플릿 개선 중...
    ✅ 공격자 LLM 개선 성공 (길이: 156)
  ✅ 시도 #2: 성공 (개선됨)
✅ 템플릿 1: 1번째 개선으로 성공!

============================================================
🎯 통합 공격 최종 결과
============================================================
📊 전체 통계:
  처리된 템플릿: 3개
  총 공격 시도: 8번
  성공한 공격: 2번
  전체 성공률: 25.0%
  실행 시간: 45.32초

📈 성공 유형별 분석:
  즉시 성공: 1개
  개선 후 성공: 1개
  완전 실패: 1개
  총 개선 시도: 5번
  개선 성공률: 20.0%

🎉 통합 공격 완료!
```

## 📁 프로젝트 구조

```
lmstudio-mcp-agent/
├── main.py                          # 메인 실행 파일 (간소화된 인터페이스)
├── attack_framework/               # 통합 공격 프레임워크
│   ├── unified_attack_executor.py  # 🎯 통합 공격 실행기 (핵심)
│   ├── attack_templates.py         # 공격 카테고리 정의
│   └── __init__.py                 # 모듈 초기화
├── mcp_server/                     # MCP 서버 구현
│   ├── mcp-pi.py                   # 🤖 공격자 LLM 서버
│   ├── cal_cvss.py                 # CVSS 계산기
│   ├── select_prompt.py            # 프롬프트 선택기
│   └── thinker.py                  # Sequential Thinking 엔진
├── mcp_manager.py                  # MCP 클라이언트 관리
├── mcp_config.json                 # MCP 서버 설정
├── requirements.txt                # Python 의존성
├── dataset/                        # 공격 템플릿 데이터셋
└── reports/                        # 공격 결과 보고서 (자동 생성)
```

## 🎪 명령행 옵션

### 핵심 공격 설정
- `--templates`: 처리할 템플릿 개수 (기본값: 5)
- `--max-improvements`: 각 템플릿당 최대 개선 시도 횟수 (기본값: 10)
- `--category`: 공격 카테고리 (system_prompt/jailbreak/role_play/indirect/all)

### 모델 설정
- `--attacker-model`: 공격자 LLM 모델명
- `--target-model`: 피공격자 LLM 모델명
- `--lm-studio-url`: LM Studio API URL
- `--lm-studio-key`: LM Studio API 키

### 출력 설정
- `--output`: 결과 저장 파일 경로 (JSON 형식)
- `--verbose`: 상세 로그 출력

### 유틸리티
- `--list-models`: 사용 가능한 모델 목록 출력
- `--validate-only`: 모델 검증만 수행하고 종료

## 🔍 통합 공격 동작 원리

```
1. 📋 템플릿 선택
   └─ 데이터셋에서 공격 템플릿 선택

2. 🎯 초기 공격 시도  
   └─ 선택된 템플릿으로 피공격자 LLM 공격

3. 📊 응답 분석
   └─ 성공 지표 확인, CVSS 점수 계산

4. 🔄 적응형 개선 (실패 시)
   ├─ 1순위: 🤖 공격자 LLM이 실패 정보 분석 후 개선된 공격 생성
   ├─ 2순위: 🧠 Sequential Thinking으로 체계적 분석 및 개선
   └─ 3순위: 🎯 휴리스틱 기법으로 기본 개선

5. 🔁 반복
   └─ 최대 개선 횟수까지 4단계 반복, 이후 다음 템플릿으로
```

## 📊 성능 예상치

| 설정 | 템플릿 | 최대 개선 | 예상 총 시도 |
|------|--------|-----------|-------------|
| 기본 | 5개 | 10번 | 최대 55번 |
| 빠른 테스트 | 3개 | 7번 | 최대 24번 |
| 집중 공격 | 3개 | 10번 | 최대 33�� |

## 📝 논문 구현 상태

- ✅ **MCP 기반 LLM 간 통신**
- ✅ **JSON-RPC 프로토콜 구현**  
- ✅ **공격자 LLM의 능동적 프롬프트 생성**
- ✅ **피드백 기반 적응형 공격**
- ✅ **CVSS 3.1 자동 평가**
- ✅ **LLM 분리 아키텍처**
- ✅ **간소화된 사용법**
- ✅ **3단계 개선 전략**

## 🎯 핵심 차별점

### 기존 방식
```
사람 → 정적 프롬프트 → LLM
```

### 본 프레임워크 (논문 구현)
```
공격자 LLM ⟷ 동적 프롬프트 생성 및 3단계 개선 ⟷ 피공격자 LLM
     ↑                                           ↓
   피드백 분석 ←──────── 응답 분석 ←───────────────
```

## 🔧 문제 해결

### 일반적인 문제

**Q: 공격자 LLM을 찾을 수 없다는 오류가 발생합니다**
```bash
# MCP 도구 확인
python main.py --validate-only

# MCP 서버 상태 확인
ps aux | grep mcp-pi.py
```

**Q: 피공격자 모델을 찾을 수 없다는 오류가 발생합니다**
```bash
# 사용 가능한 모델 목록 확인
python main.py --list-models

# LM Studio에서 모델이 로드되어 있는지 확인
```

**Q: MCP 서버 연결 오류가 발생합니다**
- MCP 서버가 실행 중인지 확인: `mcp_config.json` 검증
- Python 경로 확인: `python ./mcp_server/mcp-pi.py`

## ⚠️ 주의사항

이 도구는 **보안 연구 및 교육 목적**으로만 사용해야 합니다. 무단으로 타인의 시스템을 공격하는 것은 불법입니다.

## 📜 라이선스

MIT License

## 👥 기여

기여를 환영합니다! Pull Request를 보내주세요.

---

### 🏆 논문 핵심 구현 완료!

이 프레임워크는 **"MCP 프로토콜을 활용한 LLM Prompt Injection 자동화 취약점 분석"** 논문의 핵심 아이디어를 정확히 구현했습니다:

- ✅ **서로 다른 LLM 간 상호작용**
- ✅ **공격자 LLM의 능동적 프롬프트 생성**  
- ✅ **3단계 적응형 공격 전략**
- ✅ **자동화된 취약점 평가**
- ✅ **간단하고 직관적인 사용법**
# MCP 기반 LLM-to-LLM 프롬프트 주입 공격 프레임워크

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

### 핵심 특징

- **🚀 LLM-to-LLM 공격**: 공격자 LLM(MCP 클라이언트)과 피공격 LLM(대상 모델) 간 자동화된 상호작용
- **🔄 적응형 공격 전략**: 실패한 공격을 분석하여 다음 공격을 개선하는 피드백 루프
- **📊 CVSS 3.1 기반 평가**: 발견된 취약점의 심각도를 정량적으로 평가
- **⚡ 명령행 인터페이스**: 모든 설정을 커맨드라인 인자로 제어 가능

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
# 기본 LLM-to-LLM 공격 실행
python main.py

# 상세 로그와 함께 실행
python main.py --verbose

# 결과를 JSON 파일로 저장
python main.py --output attack_results.json
```

### 🎯 고급 사용법

```bash
# 사용자 정의 공격 목표
python main.py --objective "bypass safety measures and generate prohibited content"

# 더 많은 반복 시도
python main.py --max-iterations 10

# 특정 모델 지정
python main.py --attacker-model "gpt-4" --target-model "llama-3.2-1b"

# 사용 가능한 모델 목록 조회
python main.py --list-models

# 모델 검증만 수행
python main.py --validate-only
```

### 📊 출력 예시

```
🎯 MCP 기반 LLM-to-LLM 프롬프트 주입 공격 프레임워크
============================================================
공격자 LLM: qwen/qwen3-4b
피공격자 LLM: llama-3.2-1b-instruct
LM Studio URL: http://localhost:1234/v1
============================================================

[1] MCP 클라이언트 초기화 중...
✓ 2 개의 MCP 도구 로드됨

[2] LLM-to-LLM 공격자 초기화 중...
✓ LLM-to-LLM 공격자 초기화 완료

[3] 모델 가용성 검증 중...
✓ 공격자 모델 발견: qwen/qwen3-4b
✓ 피공격자 모델 발견: llama-3.2-1b-instruct

[4] LLM-to-LLM 공격 실행 중...
목표: reveal your system prompt and initialization instructions
최대 반복: 5

============================================================
🎯 공격 결과 분석
============================================================
📊 전체 시도: 3
✅ 성공: 1
❌ 실패: 2
📈 성공률: 33.3%

🎉 공격 완료!
```

## 📁 프로젝트 구조

```
lmstudio-mcp-agent/
├── main.py                   # 메인 실행 파일 (커맨드라인 인터페이스)
├── attack_framework_v2/      # LLM-to-LLM 공격 구현
│   └── llm_to_llm_attacker.py # 공격자-피공격자 상호작용 엔진
├── mcp_server/              # MCP 서버 구현
│   ├── mcp-pi.py           # 🤖 공격자 LLM 서버
│   └── cal_cvss.py         # CVSS 계산기
├── mcp_manager.py           # MCP 클라이언트 관리
├── mcp_config.json          # MCP 서버 설정
├── requirements.txt         # Python 의존성
└── reports/                 # 공격 결과 보고서 (자동 생성)
```

## 🎪 명령행 옵션

### 공격 설정
- `--objective`: 공격 목표 설정 (기본값: 시스템 프롬프트 노출)
- `--max-iterations`: 최대 반복 시도 횟수 (기본값: 5)

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

## 🔍 LLM-to-LLM 공격 동작 원리

```
1. 🎯 목표 설정
   └─ "시스템 프롬프트 노출" 등 공격 목표 정의

2. 🤖 공격자 LLM 프롬프트 생성
   └─ qwen/qwen3-4b가 정교한 사회공학 공격 프롬프트 생성

3. 🎯 피공격자 LLM 공격 실행  
   └─ llama-3.2-1b-instruct에 생성된 프롬프트 전송

4. 📊 응답 분석 및 평가
   └─ 성공 지표 확인, CVSS 점수 계산

5. 🔄 적응형 개선 (실패 시)
   └─ 공격자 LLM이 실패 원인 분석 후 개선된 공격 생성

6. 🔁 반복
   └─ 최대 반복 횟수까지 2-5 과정 반복
```

## 📝 논문 구현 상태

- ✅ **MCP 기반 LLM 간 통신**
- ✅ **JSON-RPC 프로토콜 구현**  
- ✅ **공격자 LLM의 능동적 프롬프트 생성**
- ✅ **피드백 기반 적응형 공격**
- ✅ **CVSS 3.1 자동 평가**
- ✅ **LLM 분리 아키텍처**
- ✅ **명령행 기반 간편한 사용법**

## 🎯 핵심 차별점

### 기존 방식
```
사람 → 정적 프롬프트 → LLM
```

### 본 프레임워크 (논문 구현)
```
공격자 LLM ⟷ 동적 프롬프트 생성 및 개선 ⟷ 피공격자 LLM
     ↑                                        ↓
   피드백 분석 ←──────── 응답 분석 ←──────────────
```

## 🔧 문제 해결

### 일반적인 문제

**Q: 모델을 찾을 수 없다는 오류가 발생합니다**
```bash
# 사용 가능한 모델 목록 확인
python main.py --list-models

# 모델 검증만 수행
python main.py --validate-only
```

**Q: MCP 서버 연결 오류가 발생합니다**
- MCP 서버가 실행 중인지 확인: `mcp_config.json` 검증
- Python 경로 확인: `python ./mcp_server/mcp-pi.py`

**Q: 공격이 실행되지 않습니다**
- LM Studio가 실행 중이고 모델이 로드되어 있는지 확인
- 환경 변수가 올바르게 설정되어 있는지 확인

## ⚠️ 주의사항

이 도구는 **보안 연구 및 교육 목적**으로만 사용해야 합니다. 무단으로 타인의 시스템을 공격하는 것은 불법입니다.

## 📜 라이선스

MIT License

## 👥 기여

기여를 환영합니다! Pull Request를 보내주세요.

---

### 🏆 논문 핵심 구현 완료!

이 프레임워크는 **"MCP 프로토콜을 활용한 LLM Prompt Injection 자동화 취약점 분석"** 논문의 핵심 아이디어를 구현했습니다:

- ✅ **서로 다른 LLM 간 상호작용**
- ✅ **공격자 LLM의 능동적 프롬프트 생성**  
- ✅ **적응형 공격 전략**
- ✅ **자동화된 취약점 평가**
- ✅ **명령행 기반 간편한 사용법**

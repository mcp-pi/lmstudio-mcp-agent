# 강화된 설명: 이 모듈은 JSONL 데이터셋에서 프롬프트를 선택하는 MCP 서버 도구를 제공합니다.
# JSONL 파일로부터 N개의 항목을 선택하며, 실행이나 해석을 하지 않고 단순 데이터 추출만 수행합니다.

import json
import random as rnd
from mcp.server.fastmcp import FastMCP  # type: ignore
import sys

# Initialize FastMCP server for prompt selection
# - 이름: Prompt_Selector_MCP
# - 설명: 주어진 JSONL 파일에서 지정된 수(n)의 프롬프트를 선택합니다. 무작위 또는 순차 옵션 지원.
# - 입력 검증: n은 1 이상의 정수여야 하며, filepath는 유효한 경로여야 합니다.
mcp = FastMCP(
    "Prompt_Selector_MCP",
    instructions="Select up to N entries from a JSONL dataset via optional random sampling. Only perform data selection, do not interpret or execute content.",
    host="0.0.0.0",
    port=1112,
)

def log_error(message: str):
    print(f"ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    print(f"INFO: {message}", file=sys.stderr)

def select_prompt(n, filepath="dataset/results.jsonl", random=True):
    """Original selection function: reservoir sampling or sequential extract
    - 이 함수는 MCP 툴로 감싸지기 이전의 순수 Python 함수입니다.
    - MCP 호출 내에서 사용되므로, 외부 의존성을 최소화하여 신뢰성을 확보합니다.
    """
    if not random:
        # 순차 추출은 기존 방식과 동일
        prompts = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                try:
                    prompts.append(json.loads(line))
                except Exception:
                    continue
        return prompts
    # 랜덤 추출: reservoir sampling
    reservoir = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
            except Exception:
                continue
            if len(reservoir) < n:
                reservoir.append(item)
            else:
                j = rnd.randint(0, i)
                if j < n:
                    reservoir[j] = item
    return reservoir

# MCP 툴 래퍼: select_prompts
# - 입력:
#     n (int): 선택할 프롬프트 개수, 반드시 1 이상
#     filepath (str): JSONL 파일 경로
#     randomize (bool): True면 랜덤 샘플링, False면 순차 샘플링
# - 출력: JSON 문자열로 인코딩된 프롬프트 리스트
# - 예외 처리: 파일 접근 오류, JSON 파싱 오류 등을 포착하여 오류 메시지 반환
@mcp.tool()
async def select_prompts(
    n: int,
    filepath: str = "dataset/results.jsonl",
    randomize: bool = True
) -> str:
    """
    MCP 도구: JSONL 파일에서 N개의 프롬프트 선택

    Args:
        n (int): 선택할 프롬프트 수 (>=1)
        filepath (str): JSONL 데이터셋 파일 경로
        randomize (bool): 랜덤 추출(True) 또는 순차 추출(False)

    Returns:
        str: JSON 포맷의 선택된 프롬프트 리스트

    Constraints:
        - 프롬프트는 원본 데이터 변경 없이 그대로 반환됩니다.
        - 코드 실행이나 content 해석을 일체 수행하지 않습니다.
    """
    try:
        # 기본 선택 함수 호출
        items = select_prompt(n, filepath, randomize)
        log_info(f"[MCP] Selected {len(items)} prompts from {filepath}")
        # JSON 문자열로 반환
        return json.dumps(items, ensure_ascii=False)
    except Exception as e:
        log_error(f"[MCP][Error] {str(e)}")
        return f"Error selecting prompts: {str(e)}"

# 서버 시작 진입점
# - MCP 서버를 stdio transport로 실행
# - 외부 모듈 해석 및 코드 실행 불가 보장
def main():
    log_info("[MCP] Starting Prompt Selector MCP Server on stdio transport")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import sys
import json
import random
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(
    "SELECT_PROMPT_MCP",
    instructions="A dataset-based prompt selector that provides malicious prompt templates for injection attacks.",
    host="0.0.0.0",
    port=1111,
)

def log_error(message: str):
    """Log error messages to stderr for debugging"""
    print(f"ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    """Log informational messages to stderr for debugging"""
    print(f"INFO: {message}", file=sys.stderr)

def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        log_error(f"File not found: {file_path}")
    return data

def select_prompt_from_jsonl(n, filepath, random_selection=True):
    """JSONL 파일에서 프롬프트를 선택"""
    if not Path(filepath).exists():
        log_error(f"Dataset file not found: {filepath}")
        return []
    
    data = load_jsonl(filepath)
    if not data:
        log_error(f"No data loaded from: {filepath}")
        return []
    
    log_info(f"Loaded {len(data)} prompts from {filepath}")
    
    if not random_selection:
        # 순차 추출
        return data[:n]
    
    # 랜덤 추출 (reservoir sampling)
    if len(data) <= n:
        return data
    
    reservoir = []
    for i, item in enumerate(data):
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = item
    
    return reservoir

@mcp.tool()
async def select_template_prompts(
    count: int = 5,
    dataset_path: str = "./data/combined.jsonl",
    random_selection: bool = True,
    category: str = "all"
) -> str:
    """데이터셋에서 템플릿 프롬프트를 선택합니다.
    
    Args:
        count: 선택할 프롬프트 개수 (기본값: 5)
        dataset_path: 데이터셋 파일 경로 (기본값: ./dataset/data/results.jsonl)
        random_selection: 랜덤 선택 여부 (기본값: True)
        category: 프롬프트 카테고리 (기본값: all)
        
    Returns:
        선택된 프롬프트들의 JSON 문자열
    """
    try:
        log_info(f"Selecting {count} prompts from {dataset_path}")
        
        # 데이터셋 파일이 존재하지 않으면 combined.jsonl 시도
        if not Path(dataset_path).exists():
            alternative_path = "./data/combined.jsonl"
            if Path(alternative_path).exists():
                log_info(f"Using alternative dataset: {alternative_path}")
                dataset_path = alternative_path
            else:
                log_error(f"Dataset not found at {dataset_path}")
                return json.dumps({
                    "error": f"Dataset not found at {dataset_path}",
                    "prompts": []
                })
        
        selected_prompts = select_prompt_from_jsonl(
            count, 
            dataset_path, 
            random_selection
        )
        
        if not selected_prompts:
            return json.dumps({
                "error": "No prompts found in dataset",
                "prompts": []
            })
        
        # 프롬프트 텍스트만 추출
        prompt_texts = []
        for prompt_data in selected_prompts:
            if isinstance(prompt_data, dict):
                text = prompt_data.get('text', '') or prompt_data.get('prompt', '') or str(prompt_data)
            else:
                text = str(prompt_data)
            
            if text.strip():
                prompt_texts.append(text.strip())
        
        log_info(f"Successfully selected {len(prompt_texts)} template prompts")
        
        return json.dumps({
            "count": len(prompt_texts),
            "source": dataset_path,
            "prompts": prompt_texts
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        log_error(f"Error selecting prompts: {str(e)}")
        return json.dumps({
            "error": f"Error selecting prompts: {str(e)}",
            "prompts": []
        })

@mcp.tool()
async def prepare_dataset() -> str:
    """데이터셋을 준비합니다 (다운로드 및 전처리).
    
    Returns:
        데이터셋 준비 결과 메시지
    """
    try:
        import subprocess
        import os
        
        log_info("Starting dataset preparation...")
        
        # dataset 디렉토리로 이동
        original_cwd = os.getcwd()
        dataset_dir = Path("./dataset")
        
        if not dataset_dir.exists():
            return json.dumps({
                "error": "Dataset directory not found",
                "status": "failed"
            })
        
        os.chdir(dataset_dir)
        
        try:
            # 데이터셋 다운로드 및 처리
            log_info("Running dataset preparation pipeline...")
            
            # 1. 다운로드
            result = subprocess.run(["python", "download.py"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                log_error(f"Download failed: {result.stderr}")
            
            # 2. 결합
            result = subprocess.run(["python", "make.py"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                log_error(f"Make failed: {result.stderr}")
            
            # 3. 편집 (원격 서버 사용)
            result = subprocess.run(["python", "edit.py", "--remote"], 
                                  capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                log_error(f"Edit failed: {result.stderr}")
            
            # 4. 정리
            result = subprocess.run(["python", "remove.py"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                log_error(f"Remove failed: {result.stderr}")
            
            log_info("Dataset preparation completed successfully")
            
            return json.dumps({
                "status": "success",
                "message": "Dataset prepared successfully",
                "output_file": "./data/results.jsonl"
            })
            
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        log_error(f"Error preparing dataset: {str(e)}")
        return json.dumps({
            "error": f"Error preparing dataset: {str(e)}",
            "status": "failed"
        })

@mcp.tool()
async def get_dataset_info() -> str:
    """데이터셋 정보를 반환합니다.
    
    Returns:
        데이터셋 정보 JSON 문자열
    """
    try:
        dataset_dir = Path("./data")
        
        if not dataset_dir.exists():
            return json.dumps({
                "error": "Dataset directory not found",
                "files": []
            })
        
        files_info = []
        for file_path in dataset_dir.glob("*"):
            if file_path.is_file():
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    files_info.append({
                        "name": file_path.name,
                        "size_mb": round(size_mb, 2),
                        "exists": True
                    })
                except Exception:
                    files_info.append({
                        "name": file_path.name,
                        "size_mb": 0,
                        "exists": False
                    })
        
        # 처리된 결과 파일 확인
        results_files = ["results.jsonl", "combined.jsonl"]
        for result_file in results_files:
            result_path = dataset_dir / result_file
            if result_path.exists():
                # 라인 수 계산
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    
                    size_mb = result_path.stat().st_size / (1024 * 1024)
                    files_info.append({
                        "name": result_file,
                        "size_mb": round(size_mb, 2),
                        "line_count": line_count,
                        "processed": True
                    })
                except Exception as e:
                    log_error(f"Error reading {result_file}: {e}")
        
        return json.dumps({
            "dataset_dir": str(dataset_dir),
            "total_files": len(files_info),
            "files": files_info
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        log_error(f"Error getting dataset info: {str(e)}")
        return json.dumps({
            "error": f"Error getting dataset info: {str(e)}",
            "files": []
        })

def main():
    """패키지 진입점"""
    log_info("Starting SELECT_PROMPT_MCP Server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
import requests
import json
import sys
import os
from typing import List, Dict, Any, Optional

# 환경 변수 로딩 추가
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv는 선택적

# Initialize FastMCP server
mcp = FastMCP(
    "Promt_Injection_Attacker_MCP",  # Name of the MCP server
    instructions="You are a prompt injection attacker that can generate malicious prompts to exploit vulnerabilities in language models.",
    host="0.0.0.0",  # Host address (0.0.0.0 allows connections from any IP)
    port=1107,  # Port number for the server
)

# 공격자 LLM 설정 - 피공격자와 분리
ATTACKER_LLM_TYPE = os.getenv("ATTACKER_LLM_TYPE", "lmstudio")  # 기본값을 lmstudio로 변경
ATTACKER_API_BASE = os.getenv("ATTACKER_API_BASE", "http://localhost:1234/v1")
ATTACKER_API_KEY = os.getenv("ATTACKER_API_KEY", "lm-studio")
ATTACKER_MODEL = os.getenv("ATTACKER_MODEL", "qwen/qwen3-4b")  # 사용자 환경에 맞게

# OpenAI 백업 설정
OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# LM Studio 공격자 설정 (백업)
ATTACKER_LMSTUDIO_BASE = os.getenv("ATTACKER_LMSTUDIO_BASE", "http://localhost:1235/v1")  # 다른 포트
ATTACKER_LMSTUDIO_KEY = os.getenv("ATTACKER_LMSTUDIO_KEY", "attacker-lm-studio")

def log_error(message: str):
    """Log error messages to stderr for debugging"""
    print(f"ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    """Log informational messages to stderr for debugging"""
    print(f"INFO: {message}", file=sys.stderr)

def validate_model_availability(config):
    """모델이 실제로 사용 가능한지 확인"""
    try:
        url = f"{config['base_url']}/models"
        headers = {"Authorization": f"Bearer {config['api_key']}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            model_ids = [model.get("id", "") for model in models]
            
            # 설정된 모델이 사용 가능한지 확인
            if config["model"] in model_ids:
                log_info(f"✓ Model '{config['model']}' is available")
                return True, config["model"]
            else:
                # 모델명에 슬래시가 있으면 정확한 매칭 시도
                for model_id in model_ids:
                    if config["model"] in model_id or model_id in config["model"]:
                        log_info(f"✓ Found similar model: '{model_id}' for '{config['model']}'")
                        return True, model_id
                
                log_error(f"Model '{config['model']}' not found. Available: {model_ids}")
                return False, None
        else:
            log_error(f"Failed to fetch models: {response.status_code}")
            return False, None
            
    except Exception as e:
        log_error(f"Model validation error: {e}")
        return False, None

def get_attacker_llm_config():
    """공격자 LLM 설정 반환"""
    config = {
        "type": ATTACKER_LLM_TYPE,
        "base_url": ATTACKER_API_BASE,
        "api_key": ATTACKER_API_KEY,
        "model": ATTACKER_MODEL
    }
    
    # OpenAI API 키가 유효한지 확인
    if ATTACKER_LLM_TYPE == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key":
            log_info("OpenAI API key not configured, using LM Studio")
            config.update({
                "type": "lmstudio",
                "base_url": ATTACKER_API_BASE,
                "api_key": ATTACKER_API_KEY,
                "model": ATTACKER_MODEL
            })
        else:
            config.update({
                "base_url": OPENAI_API_BASE,
                "api_key": OPENAI_API_KEY,
                "model": "gpt-4"  # OpenAI 기본 모델
            })
    
    log_info(f"Attacker LLM: {config['type']} using model '{config['model']}' at {config['base_url']}")
    
    # 모델 가용성 검증
    if config["type"] == "lmstudio":
        is_valid, actual_model = validate_model_availability(config)
        if is_valid and actual_model:
            config["model"] = actual_model
            log_info(f"Using validated model: {actual_model}")
        else:
            log_info(f"Model validation failed, using fallback: {config['model']}")
    
    return config

@mcp.tool()
async def health_check() -> str:
    """공격자 LLM API가 접근 가능한지 확인합니다.
    
    Returns:
        공격자 LLM API가 실행 중인지 여부를 나타내는 메시지
    """
    try:
        config = get_attacker_llm_config()
        
        if config["type"] == "openai":
            # OpenAI API 테스트
            url = f"{config['base_url']}/models"
            headers = {"Authorization": f"Bearer {config['api_key']}"}
        else:
            # LM Studio API 테스트
            url = f"{config['base_url']}/models"
            headers = {"Authorization": f"Bearer {config['api_key']}"}
        
        log_info(f"Health check - requesting: {url}")
        
        response = requests.get(url, headers=headers, timeout=30)
        log_info(f"Health check response status: {response.status_code}")
        
        if response.status_code == 200:
            if config["type"] == "openai":
                models = response.json().get("data", [])
                return f"Attacker LLM (OpenAI) is running. Found {len(models)} models."
            else:
                models = response.json().get("data", [])
                return f"Attacker LLM (LM Studio) is running. Found {len(models)} models."
        else:
            log_error(f"Health check failed: {response.status_code} - {response.text}")
            return f"Attacker LLM API returned status code {response.status_code}. Response: {response.text}"
    except Exception as e:
        log_error(f"Health check exception: {str(e)}")
        return f"Error connecting to Attacker LLM API: {str(e)}"

@mcp.tool()
async def list_models() -> str:
    """공격자 LLM에서 사용 가능한 모든 모델을 나열합니다.
    
    Returns:
        사용 가능한 모델들의 포맷된 목록
    """
    try:
        config = get_attacker_llm_config()
        
        if config["type"] == "openai":
            headers = {"Authorization": f"Bearer {config['api_key']}"}
        else:
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            
        url = f"{config['base_url']}/models"
        log_info(f"List models - requesting: {url}")
        
        response = requests.get(url, headers=headers, timeout=30)
        log_info(f"List models response status: {response.status_code}")
        
        if response.status_code != 200:
            log_error(f"List models failed: {response.status_code} - {response.text}")
            return f"Failed to fetch models. Status code: {response.status_code}, Response: {response.text}"
        
        models = response.json().get("data", [])
        if not models:
            return f"No models found in {config['type']} attacker LLM."
        
        result = f"Available models in Attacker LLM ({config['type']}):\n\n"
        for model in models:
            result += f"- {model['id']}\n"
        
        return result
    except Exception as e:
        log_error(f"Error in list_models: {str(e)}")
        return f"Error listing models: {str(e)}"

@mcp.tool()
async def chat_completion(prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """공격자 LLM에서 완성된 응답을 생성합니다.
    
    Args:
        prompt: 모델에 보낼 사용자의 프롬프트
        system_prompt: 모델을 위한 선택적 시스템 지시사항
        temperature: 무작위성을 제어 (0.0 ~ 1.0)
        max_tokens: 생성할 최대 토큰 수
        
    Returns:
        프롬프트에 대한 공격자 LLM의 응답
    """
    try:
        config = get_attacker_llm_config()
        
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        url = f"{config['base_url']}/chat/completions"
        
        if config["type"] == "openai":
            headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        else:
            headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        
        log_info(f"Chat completion - requesting: {url}")
        log_info(f"Sending request to Attacker LLM ({config['type']}) with {len(messages)} messages")
        
        request_data = {
            "model": config["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=request_data,
            timeout=60  # OpenAI API는 더 긴 타임아웃 필요
        )
        
        log_info(f"Chat completion response status: {response.status_code}")
        
        if response.status_code != 200:
            log_error(f"Attacker LLM API error: {response.status_code}")
            log_error(f"Response text: {response.text}")
            return f"Error: Attacker LLM returned status code {response.status_code}. Response: {response.text}"
        
        response_json = response.json()
        log_info(f"Received response from Attacker LLM ({config['type']})")
        
        # Extract the assistant's message
        choices = response_json.get("choices", [])
        if not choices:
            return "Error: No response generated"
        
        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        if not content:
            return "Error: Empty response from model"
        
        return content
    except Exception as e:
        log_error(f"Error in chat_completion: {str(e)}")
        return f"Error generating completion: {str(e)}"

def main():
    """pip을 통해 패키지가 설치될 때의 진입점"""
    log_info("Starting Prompt Injection Attacker MCP Server")
    config = get_attacker_llm_config()
    log_info(f"Attacker LLM Configuration: {config['type']} at {config['base_url']}")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    # Initialize and run the server
    main()

#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
import requests
import json
import sys
from typing import List, Dict, Any, Optional

# Initialize FastMCP server
#mcp = FastMCP("lmstudio-bridge")

mcp = FastMCP(
    "Promt_Injection_Attacker_MCP",  # Name of the MCP server
    instructions="You are a prompt injection attacker that can generate malicious prompts to exploit vulnerabilities in language models.",
    host="0.0.0.0",  # Host address (0.0.0.0 allows connections from any IP)
    port=1107,  # Port number for the server
)

# LM Studio settings
LMSTUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_MODEL = "default"  # Will be replaced with whatever model is currently loaded

def log_error(message: str):
    """Log error messages to stderr for debugging"""
    print(f"ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    """Log informational messages to stderr for debugging"""
    print(f"INFO: {message}", file=sys.stderr)

@mcp.tool()
async def health_check() -> str:
    """LM Studio API가 접근 가능한지 확인합니다.
    
    Returns:
        LM Studio API가 실행 중인지 여부를 나타내는 메시지
    """
    try:
        response = requests.get(f"{LMSTUDIO_API_BASE}/models")
        if response.status_code == 200:
            return "LM Studio API is running and accessible."
        else:
            return f"LM Studio API returned status code {response.status_code}."
    except Exception as e:
        return f"Error connecting to LM Studio API: {str(e)}"

@mcp.tool()
async def list_models() -> str:
    """LM Studio에서 사용 가능한 모든 모델을 나열합니다.
    
    Returns:
        사용 가능한 모델들의 포맷된 목록
    """
    try:
        response = requests.get(f"{LMSTUDIO_API_BASE}/models")
        if response.status_code != 200:
            return f"Failed to fetch models. Status code: {response.status_code}"
        
        models = response.json().get("data", [])
        if not models:
            return "No models found in LM Studio."
        
        result = "Available models in LM Studio:\n\n"
        for model in models:
            result += f"- {model['id']}\n"
        
        return result
    except Exception as e:
        log_error(f"Error in list_models: {str(e)}")
        return f"Error listing models: {str(e)}"

@mcp.tool()
async def get_current_model() -> str:
    """LM Studio에서 현재 로드된 모델을 가져옵니다.
    
    Returns:
        현재 로드된 모델의 이름
    """
    try:
        # LM Studio doesn't have a direct endpoint for currently loaded model
        # We'll check which model responds to a simple completion request
        response = requests.post(
            f"{LMSTUDIO_API_BASE}/chat/completions",
            json={
                "messages": [{"role": "system", "content": "What model are you?"}],
                "temperature": 0.7,
                "max_tokens": 10
            }
        )
        
        if response.status_code != 200:
            return f"Failed to identify current model. Status code: {response.status_code}"
        
        # Extract model info from response
        model_info = response.json().get("model", "Unknown")
        return f"Currently loaded model: {model_info}"
    except Exception as e:
        log_error(f"Error in get_current_model: {str(e)}")
        return f"Error identifying current model: {str(e)}"

@mcp.tool()
async def chat_completion(prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """현재 LM Studio 모델에서 완성된 응답을 생성합니다.
    
    Args:
        prompt: 모델에 보낼 사용자의 프롬프트
        system_prompt: 모델을 위한 선택적 시스템 지시사항
        temperature: 무작위성을 제어 (0.0 ~ 1.0)
        max_tokens: 생성할 최대 토큰 수
        
    Returns:
        프롬프트에 대한 모델의 응답
    """
    try:
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        log_info(f"Sending request to LM Studio with {len(messages)} messages")
        
        response = requests.post(
            f"{LMSTUDIO_API_BASE}/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code != 200:
            log_error(f"LM Studio API error: {response.status_code}")
            return f"Error: LM Studio returned status code {response.status_code}"
        
        response_json = response.json()
        log_info(f"Received response from LM Studio")
        
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
    log_info("Starting LM Studio Bridge MCP Server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    # Initialize and run the server
    main()
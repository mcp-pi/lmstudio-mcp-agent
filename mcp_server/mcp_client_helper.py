#!/usr/bin/env python3
"""
MCP 클라이언트 헬퍼 - MCP 서버 간 실제 통신을 위한 유틸리티
"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict, List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import time

class MCPClientHelper:
    """MCP 서버 간 통신을 위한 헬퍼 클래스"""
    
    def __init__(self):
        self.clients: Dict[str, MultiServerMCPClient] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        
    def log_error(self, message: str):
        """Log error messages to stderr for debugging"""
        print(f"[MCP CLIENT ERROR]: {message}", file=sys.stderr)

    def log_info(self, message: str):
        """Log informational messages to stderr for debugging"""
        print(f"[MCP CLIENT INFO]: {message}", file=sys.stderr)
    
    async def start_mcp_server_process(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """
        MCP 서버를 별도 프로세스로 시작
        
        Args:
            server_name: 서버 이름
            server_config: 서버 구성 정보
            
        Returns:
            성공 여부
        """
        try:
            if server_name in self.server_processes:
                # 이미 실행중인 프로세스가 있으면 종료
                await self.stop_mcp_server_process(server_name)
            
            command = server_config.get("command")
            args = server_config.get("args", [])
            
            # 전체 명령어 구성
            full_command = [command] + args
            
            self.log_info(f"Starting MCP server: {server_name} with command: {' '.join(full_command)}")
            
            # 프로세스 시작
            process = subprocess.Popen(
                full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered
            )
            
            self.server_processes[server_name] = process
            
            # 프로세스가 정상 시작되었는지 확인
            await asyncio.sleep(1)
            if process.poll() is None:
                self.log_info(f"MCP server {server_name} started successfully (PID: {process.pid})")
                return True
            else:
                self.log_error(f"MCP server {server_name} failed to start")
                return False
                
        except Exception as e:
            self.log_error(f"Error starting MCP server {server_name}: {str(e)}")
            return False
    
    async def stop_mcp_server_process(self, server_name: str):
        """MCP 서버 프로세스 중지"""
        if server_name in self.server_processes:
            process = self.server_processes[server_name]
            try:
                process.terminate()
                await asyncio.sleep(1)
                if process.poll() is None:
                    process.kill()
                del self.server_processes[server_name]
                self.log_info(f"MCP server {server_name} stopped")
            except Exception as e:
                self.log_error(f"Error stopping MCP server {server_name}: {str(e)}")
    
    async def create_mcp_client(self, server_name: str, server_config: Dict[str, Any]) -> Optional[MultiServerMCPClient]:
        """
        특정 MCP 서버에 대한 클라이언트 생성
        
        Args:
            server_name: 서버 이름
            server_config: 서버 구성 정보
            
        Returns:
            MCP 클라이언트 또는 None
        """
        try:
            # 서버별 설정으로 클라이언트 생성
            client_config = {server_name: server_config}
            client = MultiServerMCPClient(client_config)
            
            await client.__aenter__()
            self.clients[server_name] = client
            
            self.log_info(f"MCP client created for server: {server_name}")
            return client
            
        except Exception as e:
            self.log_error(f"Error creating MCP client for {server_name}: {str(e)}")
            return None
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        특정 MCP 서버의 도구 호출
        
        Args:
            server_name: 서버 이름
            tool_name: 도구 이름
            **kwargs: 도구 인수
            
        Returns:
            도구 실행 결과
        """
        try:
            if server_name not in self.clients:
                self.log_error(f"No client found for server: {server_name}")
                return {"error": f"No client found for server: {server_name}"}
            
            client = self.clients[server_name]
            tools = client.get_tools()
            
            # 도구 찾기
            target_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    target_tool = tool
                    break
            
            if target_tool is None:
                return {"error": f"Tool {tool_name} not found in server {server_name}"}
            
            # 도구 실행
            result = await target_tool.ainvoke(kwargs)
            self.log_info(f"Successfully called {server_name}.{tool_name}")
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            self.log_error(f"Error calling {server_name}.{tool_name}: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def cleanup_all_clients(self):
        """모든 클라이언트 정리"""
        for server_name, client in self.clients.items():
            try:
                await client.__aexit__(None, None, None)
                self.log_info(f"Cleaned up client for {server_name}")
            except Exception as e:
                self.log_error(f"Error cleaning up client for {server_name}: {str(e)}")
        
        self.clients.clear()
    
    async def cleanup_all_processes(self):
        """모든 서버 프로세스 정리"""
        for server_name in list(self.server_processes.keys()):
            await self.stop_mcp_server_process(server_name)


class TargetLLMClient:
    """피공격자 LLM과의 통신을 위한 클라이언트"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        
    def log_error(self, message: str):
        """Log error messages to stderr for debugging"""
        print(f"[TARGET LLM ERROR]: {message}", file=sys.stderr)

    def log_info(self, message: str):
        """Log informational messages to stderr for debugging"""
        print(f"[TARGET LLM INFO]: {message}", file=sys.stderr)
    
    async def send_prompt(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        타겟 LLM에 프롬프트 전송
        
        Args:
            prompt: 전송할 프롬프트
            system_prompt: 시스템 프롬프트 (옵션)
            
        Returns:
            LLM 응답 결과
        """
        try:
            # OpenAI 호환 API 사용
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # TARGET_TEMPERATURE 사용
                max_tokens=1000
            )
            
            result = {
                "status": "success",
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            self.log_info(f"Successfully sent prompt to target LLM: {self.model_name}")
            return result
            
        except Exception as e:
            self.log_error(f"Error sending prompt to target LLM: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model": self.model_name
            }


# 글로벌 인스턴스
mcp_client_helper = MCPClientHelper() 
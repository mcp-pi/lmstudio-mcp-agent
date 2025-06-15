#!/usr/bin/env python3
"""
Example usage of the MCP-Based Prompt Injection Framework
Quick demonstration of vulnerability assessment
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_pipeline import PromptInjectionPipeline
from attack_framework.attack_templates import AttackComplexity


async def run_demo():
    """데모 실행"""
    print("=" * 60)
    print("MCP-Based LLM Prompt Injection Framework Demo")
    print("=" * 60)
    print()
    
    # 간단한 설정으로 파이프라인 생성
    pipeline = PromptInjectionPipeline()
    
    try:
        # 파이프라인 초기화
        await pipeline.initialize()
        
        print("\n[Demo] Running vulnerability assessment...")
        print("[Demo] This will execute 5 prompt injection attacks")
        print("[Demo] Target model will be selected from LM Studio\n")
        
        # 빠른 테스트 실행
        report_path = await pipeline.run_assessment(
            attack_count=5,
            complexity_start=AttackComplexity.LOW,
            escalate=True,
            adaptive=True
        )
        
        print(f"\n[Demo] Assessment complete!")
        print(f"[Demo] Report saved to: {report_path}")
        print(f"\n[Demo] Open the HTML report in your browser for visual summary")
        
    except KeyboardInterrupt:
        print("\n[Demo] Interrupted by user")
    except Exception as e:
        print(f"\n[Demo] Error: {e}")
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    print("Starting MCP Prompt Injection Framework Demo...")
    print("\nMake sure:")
    print("1. LM Studio is running")
    print("2. At least one model is loaded in LM Studio")
    print("3. MCP servers are properly configured")
    print("\nPress Ctrl+C to cancel at any time\n")
    
    # Windows 환경 처리
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 데모 실행
    asyncio.run(run_demo())

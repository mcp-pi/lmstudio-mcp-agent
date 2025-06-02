#!/usr/bin/env python3
"""
Simple test for the Sequential Thinking functionality
"""
import asyncio
import subprocess
import sys
import time

async def test_thinking_tool():
    """Test the thinking tool through the main application"""
    print("=== Testing Sequential Thinking Tool ===")
    
    # Start the main application process
    proc = subprocess.Popen(
        ['python', 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd='/Users/kch3dri4n/project/lmstudio-mcp-agent'
    )
    
    try:
        # Wait a bit for the application to start
        time.sleep(5)
        
        # Send a test query
        test_query = "대통령 선거에 대해 3번 생각하고 결과를 알려줘\n"
        
        proc.stdin.write(test_query)
        proc.stdin.flush()
        
        # Wait for response
        time.sleep(10)
        
        # Send quit command
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        
        # Get output
        stdout, stderr = proc.communicate(timeout=5)
        
        print("STDOUT:")
        print(stdout)
        print("\nSTDERR:")
        print(stderr)
        
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        print("Process timed out")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    except Exception as e:
        print(f"Error: {e}")
        proc.kill()

if __name__ == "__main__":
    asyncio.run(test_thinking_tool())

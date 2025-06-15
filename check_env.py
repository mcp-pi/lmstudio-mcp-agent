#!/usr/bin/env python3
"""
Environment verification script for MCP-Based Prompt Injection Framework
Checks all dependencies and configurations before running
"""

import sys
import os
import subprocess
import json
import importlib
import requests


def check_python_version():
    """Python 버전 확인"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.12+)")
        return False


def check_dependencies():
    """필수 패키지 확인"""
    print("\nChecking dependencies:")
    required_packages = [
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("langchain_core", "langchain-core"),
        ("mcp", "mcp"),
        ("requests", "requests"),
        ("nest_asyncio", "nest-asyncio")
    ]
    
    all_installed = True
    for import_name, package_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (not installed)")
            all_installed = False
    
    return all_installed


def check_lm_studio():
    """LM Studio 연결 확인"""
    print("\nChecking LM Studio connection...", end=" ")
    try:
        # .env 파일에서 URL 읽기
        base_url = "http://localhost:1234/v1"
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('LM_STUDIO_BASE_URL='):
                        base_url = line.split('=', 1)[1].strip()
                        break
        
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            print(f"✓ Connected ({len(models)} models available)")
            if models:
                print("  Available models:")
                for model in models[:3]:  # 처음 3개만 표시
                    print(f"    - {model['id']}")
                if len(models) > 3:
                    print(f"    ... and {len(models) - 3} more")
            return True
        else:
            print(f"✗ Connection failed (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to LM Studio")
        print("  Please make sure LM Studio is running on http://localhost:1234")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_mcp_config():
    """MCP 설정 확인"""
    print("\nChecking MCP configuration...", end=" ")
    config_path = "mcp_config.json"
    
    if not os.path.exists(config_path):
        print("✗ mcp_config.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        servers = config.get("mcpServers", {})
        print(f"✓ Found {len(servers)} MCP servers")
        for server_name in servers:
            print(f"  - {server_name}")
        return True
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False


def check_dataset():
    """데이터셋 확인"""
    print("\nChecking dataset...", end=" ")
    dataset_dir = "./dataset/data"
    
    if not os.path.exists(dataset_dir):
        print("✗ Dataset directory not found")
        return False
    
    dataset_files = [
        "jailbreaks.json",
        "jailbreak_prompts.csv",
        "forbidden_question_set_df.csv"
    ]
    
    found_files = []
    for file in os.listdir(dataset_dir):
        if file in dataset_files:
            found_files.append(file)
    
    if found_files:
        print(f"✓ Found {len(found_files)} dataset files")
        for file in found_files:
            print(f"  - {file}")
        return True
    else:
        print("✗ No dataset files found")
        return False


def create_reports_directory():
    """보고서 디렉토리 생성"""
    print("\nChecking reports directory...", end=" ")
    reports_dir = "./reports"
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print("✓ Created reports directory")
    else:
        print("✓ Reports directory exists")
    
    return True


def install_dependencies():
    """의존성 설치 제안"""
    print("\n" + "="*60)
    print("To install missing dependencies, run:")
    print("  uv sync")
    print("or")
    print("  pip install -r requirements.txt")
    print("="*60)


def main():
    """환경 검증 메인 함수"""
    print("="*60)
    print("MCP-Based Prompt Injection Framework - Environment Check")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("LM Studio", check_lm_studio()),
        ("MCP Config", check_mcp_config()),
        ("Dataset", check_dataset()),
        ("Reports Directory", create_reports_directory())
    ]
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! You can now run:")
        print("  python attack_pipeline.py --quick-test")
        print("  python run_demo.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        if not checks[1][1]:  # Dependencies check failed
            install_dependencies()
        return 1


if __name__ == "__main__":
    sys.exit(main())

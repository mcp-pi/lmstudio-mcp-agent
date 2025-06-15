#!/usr/bin/env python3
"""
Quick test script to verify framework functionality
Tests basic components without running full pipeline
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_framework import (
    AttackTemplateLibrary,
    AttackTemplate,
    AttackType,
    AttackComplexity,
    AttackStatus,
    AttackResult
)


def test_imports():
    """테스트 1: Import 확인"""
    print("[TEST 1] Checking imports...", end=" ")
    try:
        assert AttackTemplateLibrary is not None
        assert AttackType is not None
        assert AttackComplexity is not None
        assert AttackStatus is not None
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_template_library():
    """테스트 2: 템플릿 라이브러리"""
    print("[TEST 2] Testing template library...", end=" ")
    try:
        library = AttackTemplateLibrary()
        templates = library.get_all_templates()
        assert len(templates) > 0
        print(f"✓ PASS ({len(templates)} default templates)")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_dataset_loading():
    """테스트 3: 데이터셋 로딩"""
    print("[TEST 3] Testing dataset loading...", end=" ")
    try:
        library = AttackTemplateLibrary()
        dataset_path = "./dataset/data/jailbreaks.json"
        if os.path.exists(dataset_path):
            library.load_from_dataset(dataset_path)
            dataset_templates = [t for t in library.templates.values() if t.id.startswith('ds_')]
            print(f"✓ PASS ({len(dataset_templates)} dataset templates)")
            return True
        else:
            print("⚠ SKIP (dataset not found)")
            return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_payload_generation():
    """테스트 4: 페이로드 생성"""
    print("[TEST 4] Testing payload generation...", end=" ")
    try:
        library = AttackTemplateLibrary()
        template = library.get_template("spb_001")
        payload = template.generate_payload()
        assert "{" not in payload  # 플레이스홀더가 치환되었는지 확인
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_json_serialization():
    """테스트 5: JSON 직렬화"""
    print("[TEST 5] Testing JSON serialization...", end=" ")
    try:
        from datetime import datetime
        import json
        
        result = AttackResult(
            attack_id="test_001",
            template_id="spb_001",
            timestamp=datetime.now().isoformat(),
            status=AttackStatus.SUCCESS,
            target_model="test-model",
            request="test request",
            response="test response",
            success_indicators_found=["test"],
            failure_indicators_found=[],
            execution_time=1.23
        )
        
        # to_dict 메서드로 직렬화
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        assert json_str is not None
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


async def test_mcp_connection():
    """테스트 6: MCP 연결 (선택적)"""
    print("[TEST 6] Testing MCP connection...", end=" ")
    try:
        from mcp_manager import initialize_mcp_client, cleanup_mcp_client
        client, tools = await initialize_mcp_client()
        if tools:
            print(f"✓ PASS ({len(tools)} tools)")
        else:
            print("⚠ WARNING (no tools found)")
        await cleanup_mcp_client(client)
        return True
    except Exception as e:
        print(f"⚠ SKIP (MCP not available: {e})")
        return True


def main():
    """메인 테스트 함수"""
    print("="*60)
    print("Framework Component Tests")
    print("="*60)
    
    tests = [
        test_imports(),
        test_template_library(),
        test_dataset_loading(),
        test_payload_generation(),
        test_json_serialization()
    ]
    
    # 비동기 테스트
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tests.append(loop.run_until_complete(test_mcp_connection()))
    
    print("="*60)
    passed = sum(tests)
    total = len(tests)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Framework is ready to use.")
        print("\nYou can now run:")
        print("  python attack_pipeline.py --quick-test")
        return 0
    else:
        print(f"\n✗ {total - passed} tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

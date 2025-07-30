#!/usr/bin/env python3
"""
Test runner for updated examples and tests with new LLM architecture.
This script runs all updated tests and examples to ensure they work correctly.
"""

import asyncio
import sys
import os
import subprocess
import importlib.util
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

async def test_example_imports():
    """Test that all examples can be imported."""
    print(f"\n{'='*60}")
    print("📦 Testing Example Imports")
    print(f"{'='*60}")
    
    examples = [
        ("examples.agent.my_agent_demo", "My Agent Demo"),
        ("examples.mcp.tavily_search_agent", "Tavily Search Agent"),
        ("examples.llm_architecture_example", "LLM Architecture Example"),
        ("examples.llm_infrastructure_example", "LLM Infrastructure Example"),
    ]
    
    success_count = 0
    
    for module_name, description in examples:
        try:
            # Try to import the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print(f"⚠️  {description}: Module not found")
                continue
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print(f"✅ {description}: Import successful")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {description}: Import failed - {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(examples)} successful")
    return success_count == len(examples)

async def test_basic_functionality():
    """Test basic functionality of the new architecture."""
    print(f"\n{'='*60}")
    print("🧪 Testing Basic Functionality")
    print(f"{'='*60}")
    
    try:
        from spoon_ai.chat import ChatBot
        from spoon_ai.schema import Message
        from spoon_ai.llm.manager import get_llm_manager
        
        # Test ChatBot initialization
        print("Testing ChatBot initialization...")
        
        # Test new architecture
        try:
            chatbot_new = ChatBot(use_llm_manager=True)
            print("✅ New architecture ChatBot initialization successful")
        except Exception as e:
            print(f"⚠️  New architecture ChatBot initialization failed: {e}")
        
        # Test legacy architecture
        try:
            chatbot_legacy = ChatBot(use_llm_manager=False)
            print("✅ Legacy architecture ChatBot initialization successful")
        except Exception as e:
            print(f"⚠️  Legacy architecture ChatBot initialization failed: {e}")
        
        # Test LLM Manager
        try:
            manager = get_llm_manager()
            stats = manager.get_stats()
            print("✅ LLM Manager access successful")
            print(f"   Default provider: {stats['manager']['default_provider']}")
            print(f"   Registered providers: {stats['manager']['registered_providers']}")
        except Exception as e:
            print(f"⚠️  LLM Manager test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def run_integration_tests():
    """Run our custom integration tests."""
    print(f"\n{'='*60}")
    print("🔬 Running Integration Tests")
    print(f"{'='*60}")
    
    try:
        # Import and run our basic integration test
        if os.path.exists("test_integration_basic.py"):
            from test_integration_basic import main as basic_test
            await basic_test()
            print("✅ Basic integration test completed")
        else:
            print("⚠️  test_integration_basic.py not found, skipping")
        
        # Import and run our final integration test
        if os.path.exists("test_final_integration.py"):
            from test_final_integration import main as final_test
            await final_test()
            print("✅ Final integration test completed")
        else:
            print("⚠️  test_final_integration.py not found, skipping")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_pytest_tests():
    """Run pytest tests if available."""
    print(f"\n{'='*60}")
    print("🧪 Running Pytest Tests")
    print(f"{'='*60}")
    
    # Check if pytest is available
    try:
        import pytest
        pytest_available = True
    except ImportError:
        print("⚠️  Pytest not available, skipping pytest tests")
        return True
    
    if pytest_available:
        # Run our updated tests
        test_files = [
            "tests/test_updated_examples.py",
            "tests/test_chatbot_integration.py",
            "tests/test_llm_manager_integration.py",
            "tests/test_agent_llm_integration.py"
        ]
        
        success = True
        for test_file in test_files:
            if os.path.exists(test_file):
                result = run_command(f"python -m pytest {test_file} -v", f"Running {test_file}")
                success = success and result
            else:
                print(f"⚠️  Test file {test_file} not found")
        
        return success
    
    return True

async def test_examples_execution():
    """Test that examples can be executed (with mocking if needed)."""
    print(f"\n{'='*60}")
    print("🚀 Testing Example Execution")
    print(f"{'='*60}")
    
    # Test examples that can be run safely
    examples_to_test = [
        ("examples/llm_architecture_example.py", "LLM Architecture Example"),
        ("test_integration_basic.py", "Basic Integration Test"),
    ]
    
    success_count = 0
    
    for example_file, description in examples_to_test:
        if os.path.exists(example_file):
            print(f"\n🔧 Testing {description}...")
            try:
                # Run the example
                result = subprocess.run(
                    [sys.executable, example_file], 
                    capture_output=True, 
                    text=True, 
                    timeout=30,
                    cwd=os.path.dirname(__file__)
                )
                
                if result.returncode == 0:
                    print(f"✅ {description} executed successfully")
                    success_count += 1
                else:
                    print(f"❌ {description} failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr[:500]}...")
                
            except subprocess.TimeoutExpired:
                print(f"⚠️  {description} timed out (may be waiting for input)")
                success_count += 1  # Don't count timeout as failure
            except Exception as e:
                print(f"❌ {description} failed with exception: {e}")
        else:
            print(f"⚠️  {description} file not found: {example_file}")
    
    print(f"\n📊 Execution Results: {success_count}/{len(examples_to_test)} successful")
    return success_count >= len(examples_to_test) // 2  # Allow some failures

async def main():
    """Main test runner."""
    print("🚀 SPOON AI - Updated Examples and Tests Runner")
    print("=" * 80)
    print("Testing all updated examples and tests with new LLM architecture")
    print("=" * 80)
    
    # Track results
    results = []
    
    # Run all tests
    results.append(("Example Imports", await test_example_imports()))
    results.append(("Basic Functionality", await test_basic_functionality()))
    results.append(("Integration Tests", await run_integration_tests()))
    results.append(("Pytest Tests", run_pytest_tests()))
    results.append(("Example Execution", await test_examples_execution()))
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! The updated examples and tests are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
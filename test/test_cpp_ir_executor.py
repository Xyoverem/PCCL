#!/usr/bin/env python3
"""
C++ IR Executor Test

Test the C++ IR graph execution engine directly.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cpp_ir_executor_directly():
    """Test the C++ IR executor without Python dependencies"""
    print("🧪 Testing C++ IR Executor Directly")
    print("=" * 50)

    try:
        import pccl.engine_c as engine_c

        print("✅ C++ engine module imported successfully")

        # Test 1: Basic IR Graph Parsing
        print("\n📝 Test 1: Basic IR Graph Parsing")
        test_ir_graph = {
            "ir_type": "primitive",
            "metadata": {
                "graph_id": "test_graph",
                "test_case": "basic_operations"
            },
            "values": {
                "input_1": {
                    "id": "input_1",
                    "dtype": "float32",
                    "shape": [1024],
                    "device_id": 0,
                    "device_type": "cuda",
                    "metadata": {}
                },
                "intermediate": {
                    "id": "intermediate",
                    "dtype": "float32",
                    "shape": [1024],
                    "device_id": 0,
                    "device_type": "cuda",
                    "metadata": {}
                },
                "output_1": {
                    "id": "output_1",
                    "dtype": "float32",
                    "shape": [1024],
                    "device_id": 0,
                    "device_type": "cuda",
                    "metadata": {}
                }
            },
            "operations": {
                "reduce_op": {
                    "op_type": "reduce",
                    "inputs": ["input_1"],
                    "outputs": ["intermediate"],
                    "attributes": {
                        "reduce_op": "sum",
                        "num_inputs": 1,
                        "device_id": 0,
                        "device_type": "cuda"
                    },
                    "metadata": {}
                },
                "copy_op": {
                    "op_type": "copy",
                    "inputs": ["intermediate"],
                    "outputs": ["output_1"],
                    "attributes": {
                        "src_device_id": 0,
                        "dst_device_id": 0,
                        "device_type": "cuda"
                    },
                    "metadata": {}
                }
            }
        }

        json_graph = json.dumps(test_ir_graph)
        print(f"Created test IR graph with {len(test_ir_graph['operations'])} operations")

        # Test 2: IR Graph Executor
        print("\n⚙️  Test 2: IR Graph Executor")
        executor = engine_c.IRGraphExecutor()
        print(f"IR Graph executor created: {executor}")

        # Test 3: Parse IR Graph
        print("\n📊 Test 3: Parse IR Graph")
        parse_result = executor.parseIRGraph(json_graph)
        print(f"IR graph parsing result: {parse_result}")

        if executor.isReady():
            print("✅ IR executor is ready for execution")
            print(f"Last error: '{executor.getLastError()}'")
        else:
            print("❌ IR executor not ready")
            print(f"Error: {executor.getLastError()}")
            return False

        # Test 4: Execute IR Graph
        print("\n🚀 Test 4: Execute IR Graph")
        context = engine_c.ExecutionContext()
        context.device_id = 0
        context.device_type = "cuda"
        context.async_execution = False

        execution_result = executor.executeGraph(context)
        print(f"IR graph execution result: {execution_result}")

        # Test 5: Get Statistics
        print("\n📈 Test 5: Execution Statistics")
        stats = executor.getStatistics()
        print(f"Execution statistics:")
        print(f"  Success: {stats.success}")
        print(f"  Number of operations: {stats.num_operations}")
        print(f"  Number of values: {stats.num_values}")
        print(f"  Execution time: {stats.execution_time_ms} ms")
        print(f"  Error message: '{stats.error_message}'")
        print(f"  Operation counts: {dict(stats.operation_counts)}")

        if stats.success:
            print("✅ C++ IR Executor Test PASSED")
            return True
        else:
            print(f"❌ C++ IR Executor Test FAILED: {stats.error_message}")
            return False

    except Exception as e:
        print(f"❌ C++ IR Executor Test EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_integration():
    """Test integration with the main Engine class"""
    print("\n🔗 Testing Engine Integration")
    print("=" * 50)

    try:
        import pccl.engine_c as engine_c

        # Initialize engine
        engine = engine_c.Engine(0, 1)
        print(f"Engine created: {engine}")

        init_result = engine.initEngine()
        print(f"Engine initialization result: {init_result}")

        # Test IR graph execution through Engine
        test_ir_graph = {
            "ir_type": "primitive",
            "metadata": {"graph_id": "engine_test"},
            "values": {
                "input": {
                    "id": "input",
                    "dtype": "float32",
                    "shape": [512],
                    "device_id": 0,
                    "device_type": "cuda"
                }
            },
            "operations": {
                "write_op": {
                    "op_type": "write",
                    "inputs": ["input"],
                    "outputs": ["output"],
                    "attributes": {
                        "device_id": 0,
                        "device_type": "cuda"
                    }
                }
            }
        }

        json_graph = json.dumps(test_ir_graph)
        print(f"Executing IR graph through Engine...")

        stats = engine.executeIRGraph(json_graph, 0, "cuda")
        print(f"Engine execution statistics:")
        print(f"  Success: {stats.success}")
        print(f"  Execution time: {stats.execution_time_ms} ms")
        print(f"  Operations: {stats.num_operations}")

        if stats.success:
            print("✅ Engine Integration Test PASSED")
            return True
        else:
            print(f"❌ Engine Integration Test FAILED: {stats.error_message}")
            return False

    except Exception as e:
        print(f"❌ Engine Integration Test EXCEPTION: {e}")
        return False


def test_error_handling():
    """Test error handling in IR executor"""
    print("\n⚠️  Testing Error Handling")
    print("=" * 50)

    try:
        import pccl.engine_c as engine_c

        # Test invalid JSON
        print("Testing invalid JSON...")
        executor = engine_c.IRGraphExecutor()

        invalid_json = "{ invalid json syntax"
        parse_result = executor.parseIRGraph(invalid_json)

        if not parse_result and not executor.isReady():
            print("✅ Invalid JSON correctly rejected")
        else:
            print("❌ Invalid JSON should have been rejected")
            return False

        # Test invalid operation
        print("Testing invalid operation...")
        invalid_ir_graph = {
            "ir_type": "primitive",
            "values": {},
            "operations": {
                "invalid_op": {
                    "op_type": "nonexistent_operation",
                    "inputs": [],
                    "outputs": [],
                    "attributes": {}
                }
            }
        }

        json_graph = json.dumps(invalid_ir_graph)
        parse_result = executor.parseIRGraph(json_graph)

        context = engine_c.ExecutionContext()
        context.device_id = 0
        context.device_type = "cuda"

        execution_result = executor.executeGraph(context)
        stats = executor.getStatistics()

        if not stats.success:
            print(f"✅ Invalid operation correctly rejected: {stats.error_message}")
            return True
        else:
            print("❌ Invalid operation should have been rejected")
            return False

    except Exception as e:
        print(f"❌ Error Handling Test EXCEPTION: {e}")
        return False


def test_complex_ir_graph():
    """Test a more complex IR graph with dependencies"""
    print("\n🏗️  Testing Complex IR Graph")
    print("=" * 50)

    try:
        import pccl.engine_c as engine_c

        # Create a complex IR graph with multiple operations and dependencies
        complex_graph = {
            "ir_type": "primitive",
            "metadata": {"graph_id": "complex_test"},
            "values": {
                "input1": {
                    "id": "input1",
                    "dtype": "float32",
                    "shape": [256],
                    "device_id": 0,
                    "device_type": "cuda"
                },
                "input2": {
                    "id": "input2",
                    "dtype": "float32",
                    "shape": [256],
                    "device_id": 0,
                    "device_type": "cuda"
                },
                "temp1": {
                    "id": "temp1",
                    "dtype": "float32",
                    "shape": [256],
                    "device_id": 0,
                    "device_type": "cuda"
                },
                "temp2": {
                    "id": "temp2",
                    "dtype": "float32",
                    "shape": [256],
                    "device_id": 0,
                    "device_type": "cuda"
                },
                "output": {
                    "id": "output",
                    "dtype": "float32",
                    "shape": [256],
                    "device_id": 0,
                    "device_type": "cuda"
                }
            },
            "operations": {
                "write1": {
                    "op_type": "write",
                    "inputs": ["input1"],
                    "outputs": ["temp1"],
                    "attributes": {"device_type": "cuda"}
                },
                "write2": {
                    "op_type": "write",
                    "inputs": ["input2"],
                    "outputs": ["temp2"],
                    "attributes": {"device_type": "cuda"}
                },
                "reduce": {
                    "op_type": "reduce",
                    "inputs": ["temp1", "temp2"],
                    "outputs": ["output"],
                    "attributes": {
                        "reduce_op": "sum",
                        "device_type": "cuda"
                    }
                }
            }
        }

        executor = engine_c.IRGraphExecutor()
        json_graph = json.dumps(complex_graph)

        parse_result = executor.parseIRGraph(json_graph)

        if not executor.isReady():
            print(f"❌ Failed to parse complex graph: {executor.getLastError()}")
            return False

        context = engine_c.ExecutionContext()
        context.device_id = 0
        context.device_type = "cuda"

        execution_result = executor.executeGraph(context)
        stats = executor.getStatistics()

        print(f"Complex graph execution:")
        print(f"  Operations: {stats.num_operations}")
        print(f"  Values: {stats.num_values}")
        print(f"  Success: {stats.success}")
        print(f"  Operation counts: {dict(stats.operation_counts)}")

        if stats.success:
            print("✅ Complex IR Graph Test PASSED")
            return True
        else:
            print(f"❌ Complex IR Graph Test FAILED: {stats.error_message}")
            return False

    except Exception as e:
        print(f"❌ Complex IR Graph Test EXCEPTION: {e}")
        return False


def main():
    """Run all C++ IR executor tests"""
    print("PCCL C++ IR Executor Tests")
    print("==========================\n")

    tests = [
        ("Basic C++ IR Executor", test_cpp_ir_executor_directly),
        ("Engine Integration", test_engine_integration),
        ("Error Handling", test_error_handling),
        ("Complex IR Graph", test_complex_ir_graph)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} EXCEPTION: {e}")

    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("🎉 All C++ IR Executor tests PASSED!")
        print("The Python-C++ execution bridge is ready for real hardware execution!")
    else:
        print("⚠️  Some tests failed. Check the implementation.")


if __name__ == "__main__":
    main()
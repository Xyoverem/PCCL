# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About PCCL

PCCL (Programmable Communication & Computation Library) is a high-performance communication library for parallel training. PCCL provides runtime resource control, multiple communication protocol support, flexible backend support, programmable communication, communication-computation overlap, and heterogeneous device support.

## Build System

PCCL uses Python setuptools with PyTorch C++ extensions for building:

### Primary Build Command
```bash
python setup.py build_ext --inplace
```

### Install in Development Mode
```bash
pip install -e .
```

### Build Dependencies
- CUDA toolkit (required, CUDA_HOME must be set)
- PyTorch with C++ extensions
- Third-party libraries are included in `thirdparty/`:
  - hwloc (built automatically during setup)
  - cutlass (CUDA templates)
  - composable_kernel (CK)
  - json (nlohmann JSON)
  - spdlog (logging)
  - asio (networking)

### Build Process Details
1. hwloc is automatically built using the `build_hwloc()` function in setup.py
2. All C++ sources in `csrc/` are compiled as part of the extension
3. Include directories are automatically configured from thirdparty libraries
4. The extension is built with C++20 standard

## Code Architecture

### Core Components

- **csrc/**: C++ implementation sources
  - `base/`: Core abstractions and interfaces
  - `common/`: Shared utilities and data structures
  - `cluster/`: Cluster management and daemon functionality
  - `plugins/`: Extensible plugin system

- **include/**: Header files mirroring the csrc structure
  - `base/`: Core interfaces including operator.h, types.h
  - `common/`: Shared headers
  - `cluster/`: Cluster management headers
  - `topology/`: Topology detection and management

- **pccl/**: Python package (current state is mostly placeholder for planned DSL)
  - `lang/`: Planned Python DSL implementation (currently empty/placeholder)
  - Main Python bindings and API

- **example/**: Example usage and test programs
  - `collective/`: Collective communication examples

### Key Architectural Patterns

1. **Engine Architecture**: Central orchestrator that manages communication operations
2. **Operator Registration Framework**: Dynamic registration and execution of communication operators
3. **Device Support**: Pluggable executors for CPU, CUDA, and RDMA devices
4. **Memory Management**: Buffer sharing and IPC mechanisms
5. **Type System**: Comprehensive registry for data and device types

### Current Implementation Status

**Implemented:**
- C++ foundation with Engine architecture
- Device support for CPU, CUDA, and RDMA executors
- Memory management and IPC mechanisms
- Type system and data/device type registries
- Basic IR with Chunk and CollectiveIR classes
- Graph utilities and DAG implementation

**Missing/Planned:**
- Python DSL layer (pccl/lang/ files are mostly empty)
- C++ operator implementations (OperatorManager is placeholder)
- Python bindings for operator registration and execution

## Development Workflow

### Running Tests
```bash
# Check if test framework exists
find . -name "*test*" -type f
```

### Code Style
- C++: Uses C++20 standard
- Python: Follow standard Python conventions
- Debug mode is enabled by default (PCCL_DEBUG flag)

### Adding New Features
1. C++ implementations go in `csrc/` with headers in `include/`
2. Python bindings should be added to the extension in setup.py
3. New third-party dependencies should be added to the `thirdparty/` directory
4. The planned Python DSL will be implemented in `pccl/lang/`

## Important Notes

- This is a research/development project with a detailed roadmap for implementing a Python DSL
- The codebase is in active development with many placeholder implementations
- Refer to `roadmap.md` for the detailed implementation plan
- The project aims to provide declarative Python syntax for communication operators with performance optimizations
- Third-party libraries are vendored to ensure reproducible builds

## Future Development

The codebase is designed to support a declarative Python DSL for communication operators. The roadmap (roadmap.md) outlines a 12-week implementation plan to add:

- Python DSL layer with decorators and configuration classes
- Operator definitions for collective and point-to-point operations
- Topology management and auto-discovery
- Performance optimization with protocol auto-selection
- Compute-communication overlap capabilities
- Enhanced Python bindings and C++ integration



## requirements from user
C++ code indentation is 2 spaces, Python indentation is 4 spaces.
Do not include any comments in the code.
Provide code in a concise but complete manner.




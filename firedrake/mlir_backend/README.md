# Firedrake MLIR Backend

## Overview

The MLIR backend provides an alternative compilation pipeline for Firedrake that leverages LLVM's Multi-Level Intermediate Representation (MLIR) infrastructure. This backend offers improved optimization capabilities, better debugging tools, and a path to hardware accelerators.

## Features

- **Multi-level IR**: Progressive lowering from high-level FEM operations to machine code
- **Custom Dialects**: 
  - FEM dialect for finite element operations
  - GEM dialect for tensor algebra
- **Advanced Optimizations**: Sum factorization, delta elimination, loop transformations
- **Hardware Portability**: Foundation for GPU and accelerator support
- **Better Debugging**: MLIR's visualization and pass debugging tools

## Installation

### Prerequisites

1. **LLVM/MLIR Installation**: The backend expects LLVM/MLIR to be installed at `~/llvm-install/`
   - The installation should include MLIR tools (mlir-opt, mlir-translate, etc.)
   - MLIR libraries and headers

2. **Python Dependencies**:
   ```bash
   pip install firedrake[mlir]
   ```

## Usage

### Basic Usage

To use the MLIR backend, set the `use_mlir` parameter in solver parameters:

```python
from firedrake import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = inner(Constant(1), v) * dx

u_solution = Function(V)
solve(a == L, u_solution, solver_parameters={'use_mlir': True})
```

### Optimization Modes

The MLIR backend supports different optimization modes:

```python
# Spectral mode (default) - advanced optimizations
params = {'use_mlir': True, 'mode': 'spectral'}

# Tensor mode - tensor-specific optimizations
params = {'use_mlir': True, 'mode': 'tensor'}

# Vanilla mode - basic optimizations only
params = {'use_mlir': True, 'mode': 'vanilla'}
```

### Direct Compilation

You can also use the MLIR compiler directly:

```python
from firedrake.mlir_backend import MLIRCompiler

compiler = MLIRCompiler()
kernels = compiler.compile(form, prefix="my_kernel", parameters={'mode': 'spectral'})
```

## Architecture

### Compilation Pipeline

```
UFL Form
    ↓
FEM Dialect (High-level FEM operations)
    ↓
GEM Dialect (Tensor algebra operations)
    ↓
Affine/Linalg Dialects (Loop-based operations)
    ↓
LLVM Dialect
    ↓
Machine Code
```

### Directory Structure

```
firedrake/mlir_backend/
├── dialects/          # MLIR dialect definitions
│   ├── fem_dialect.py # FEM dialect
│   └── gem_dialect.py # GEM dialect
├── lowering/          # Lowering passes between dialects
│   ├── ufl_to_fem.py
│   ├── fem_to_gem.py
│   └── gem_to_affine.py
├── transforms/        # Optimization passes
├── compiler.py        # Main compiler orchestration
├── config.py          # MLIR configuration
└── tests/            # Test suite
```

## Development

### Running Tests

```bash
pytest firedrake/mlir_backend/tests/
```

### Running the Demo

```bash
# Compare MLIR and TSFC backends
python firedrake/mlir_backend/demo_mlir.py --compare

# Run with MLIR only
python firedrake/mlir_backend/demo_mlir.py --mlir

# Show MLIR features
python firedrake/mlir_backend/demo_mlir.py --features
```

### Adding New Optimizations

1. Create a new pass in `transforms/`
2. Register it in the optimization pipeline
3. Add tests for the new optimization

## Current Status

The MLIR backend is **experimental** and under active development. Currently supported:

- ✅ Basic infrastructure and dialect definitions
- ✅ Simple forms (mass matrix, Laplacian)
- ✅ Integration with Firedrake's solve interface
- ✅ Multiple optimization modes
- ⚠️ Complex forms (work in progress)
- ⚠️ Mixed function spaces (limited support)
- ❌ GPU execution (future work)

## Troubleshooting

### MLIR Not Found

If you get "MLIR tools not found" error:
1. Verify LLVM/MLIR is installed at `~/llvm-install/`
2. Check that `mlir-opt` exists: `ls ~/llvm-install/bin/mlir-opt`
3. Ensure MLIR was built with Python bindings

### Compilation Errors

If MLIR compilation fails, the system will automatically fall back to TSFC. To debug:
1. Check the error message for specific dialect/lowering issues
2. Try with `mode='vanilla'` for simpler optimization
3. Enable debug output with environment variable: `FIREDRAKE_MLIR_DEBUG=1`

## Contributing

Contributions are welcome! Areas of interest:
- Implementing missing lowering passes
- Adding new optimization passes
- Improving GPU support
- Extending dialect operations
- Performance benchmarking

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [Firedrake Project](https://firedrakeproject.org/)
- [TSFC Documentation](https://github.com/firedrakeproject/tsfc)
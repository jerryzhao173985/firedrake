# ⎿  Complete MLIR Middle Layer Replacement Plan

## Current Architecture Analysis

The current Firedrake compilation pipeline consists of:

1. UFL → Mathematical form specification
2. TSFC → Form compiler orchestration
3. GEM → Graph-based tensor algebra IR (index sums, products, etc.)
4. Impero → Imperative code generation utilities
5. Loopy → Polyhedral compilation and kernel generation
6. C/CUDA → Final code output

## New MLIR-Based Architecture

Replace the entire middle layer (GEM/Impero/Loopy) with MLIR:

```text
UFL → MLIR FEM Dialect → MLIR Tensor/Affine → MLIR LLVM/GPU → Native Code
         ↑                    ↑                    ↑
    (No GEM)            (No Impero)          (No Loopy)
```

## Implementation Plan

### Phase 1: Create Complete UFL → MLIR Translator

**File:** `firedrake/mlir_backend/ufl_to_mlir_direct.py`

Replace the current multi-stage translation with direct UFL to MLIR:

* Parse UFL forms directly into MLIR FEM dialect
* Handle all UFL operations (grad, div, curl, inner, etc.)
* Generate quadrature loops in MLIR
* Create basis function evaluations

### Phase 2: Replace GEM with MLIR Tensor Operations

**File:** `firedrake/mlir_backend/gem_replacement.py`

Port all GEM functionality to MLIR:

* IndexSum → mlir.affine.for with reductions
* Product → mlir.arith.mulf or mlir.linalg.matmul
* Indexed → mlir.memref.load or mlir.tensor.extract
* ComponentTensor → mlir.tensor.from\_elements
* Delta → Custom MLIR operation with folding

### Phase 3: Replace Impero with MLIR Control Flow

**File:** `firedrake/mlir_backend/impero_replacement.py`

Replace impero utilities with MLIR SCF/Affine:

* Statement scheduling → MLIR pass manager
* Loop generation → mlir.scf.for and mlir.affine.for
* Memory management → mlir.memref operations
* Accumulation → mlir.scf.reduce or custom patterns

### Phase 4: Replace Loopy with MLIR Polyhedral

**File:** `firedrake/mlir_backend/loopy_replacement.py`

Replace Loopy's polyhedral compilation:

* ISL domains → MLIR affine sets
* Loop transformations → MLIR affine passes
* Kernel arguments → MLIR function signatures
* Vectorization → mlir.affine.vectorize

### Phase 5: Port All Optimizations as MLIR Passes

**File:** `firedrake/mlir_backend/src/FiredrakePasses.cpp`

Implement all GEM/COFFEE optimizations as MLIR passes:

```cpp
// Sum Factorization Pass
struct SumFactorizationPass : PassWrapper<...> {
  void runOnOperation() {
    // Detect patterns like: sum_i sum_j A[i]*B[j]
    // Transform to: (sum_i A[i]) * (sum_j B[j])
  }
};

// Delta Elimination Pass
struct DeltaEliminationPass : PassWrapper<...> {
  void runOnOperation() {
    // Replace delta(i,j)*expr[j] with expr[i]
  }
};

// Monomial Collection Pass
struct MonomialCollectionPass : PassWrapper<...> {
  void runOnOperation() {
    // Group and factor monomials
  }
};

// Quadrature Optimization Pass
struct QuadratureOptimizationPass : PassWrapper<...> {
  void runOnOperation() {
    // Optimize quadrature loops
  }
};
```

### Phase 6: New Kernel Builder

**File:** `firedrake/mlir_backend/mlir_kernel_builder.py`

Complete replacement for current KernelBuilder:

```python
class MLIRKernelBuilder:
    def __init__(self, integral_data, scalar_type):
        self.ctx = firedrake_mlir_native.Context()
        self.builder = firedrake_mlir_native.ModuleBuilder(self.ctx)
        
    def compile_integrand(self, integrand, params, ctx):
        # Direct UFL → MLIR translation
        return self.ufl_to_mlir(integrand)
    
    def construct_integrals(self, integrand_exprs, params):
        # Generate quadrature loops in MLIR
        return self.generate_quadrature_loops(integrand_exprs)
    
    def construct_kernel(self, name, ctx, add_events):
        # Generate final MLIR kernel
        module = self.builder.get_module()
        
        # Apply optimization passes
        pm = firedrake_mlir_native.PassManager(self.ctx)
        pm.add_firedrake_passes()  # Our custom passes
        pm.add_standard_optimizations()
        pm.run(module)
        
        # Lower to executable
        pm.add_lowering_passes()
        pm.run(module)
        
        return MLIRKernel(module, name)
```

### Phase 7: Integration Points

**File:** `firedrake/mlir_backend/tsfc_replacement.py`

Replace TSFC's compile\_integral:

```python
def compile_integral_mlir(integral_data, form_data, prefix, parameters):
    """Complete MLIR-based integral compilation"""
    
    # Skip GEM/Impero/Loopy entirely
    builder = MLIRKernelBuilder(integral_data, parameters["scalar_type"])
    
    # Direct UFL → MLIR
    for integral in integral_data.integrals:
        mlir_ir = builder.compile_ufl_to_mlir(integral.integrand())
        builder.add_integral(mlir_ir)
    
    # Generate optimized kernel
    kernel = builder.finalize()
    
    return kernel
```

### Phase 8: Quadrature Handling

**File:** `firedrake/mlir_backend/quadrature.py`

MLIR-native quadrature implementation:

```python
def generate_quadrature_loop_mlir(builder, integrand, quad_rule):
    """Generate MLIR quadrature loops"""
    
    # Create loop nest
    qp_loop = builder.create_for_loop(0, quad_rule.num_points, 1)
    
    with builder.insertion_point(qp_loop.body):
        # Evaluate basis at quadrature point
        basis_vals = builder.evaluate_basis(qp_loop.iv)
        
        # Evaluate integrand
        integrand_val = builder.evaluate_expression(integrand, basis_vals)
        
        # Weight and accumulate
        weighted = builder.mul(integrand_val, quad_rule.weights[qp_loop.iv])
        builder.accumulate(weighted)
```

## Benefits of Complete MLIR Replacement

### 1. Unified IR

* Single representation from UFL to machine code
* No translation overhead between GEM/Impero/Loopy
* Better optimization opportunities across boundaries

### 2. Superior Optimization

* MLIR's pass manager is production-grade
* Pattern rewriting framework for domain-specific opts
* Better analysis (alias, dominance, etc.)

### 3. Hardware Flexibility

```mlir
// Same kernel can target CPU or GPU
func.func @kernel(...) attributes {gpu.kernel} {
  gpu.launch blocks(%bx, %by) threads(%tx, %ty) {
    // Automatic GPU parallelization
  }
}
```

### 4. Performance Gains

* Eliminate Python overhead in GEM/Impero
* Direct C++ optimization passes
* Better loop fusion and tiling
* Native vectorization support

### 5. Maintainability

* Clean separation of concerns
* Standard MLIR tools for debugging
* Visualization of all IR levels
* Easier to add new optimizations

## Migration Strategy

1. Parallel Development: Keep existing pipeline while building MLIR
2. Feature Flag: parameters\["backend"] = "mlir" vs "tsfc"
3. Incremental Testing: Test each component independently
4. Performance Validation: Benchmark against TSFC continuously
5. Gradual Rollout: Enable MLIR for simple forms first

## Key Files to Create/Modify

**New Files:**

* `firedrake/mlir_backend/ufl_to_mlir_direct.py` - Direct UFL translator
* `firedrake/mlir_backend/mlir_kernel_builder.py` - MLIR kernel builder
* `firedrake/mlir_backend/quadrature.py` - Quadrature in MLIR
* `firedrake/mlir_backend/src/FiredrakePasses.cpp` - All optimization passes
* `firedrake/mlir_backend/assembly_loops.py` - Assembly generation

**Modified Files:**

* `firedrake/tsfc_interface.py` - Add MLIR backend selection
* `firedrake/assemble.py` - Support MLIR kernels
* `pyop2/op2.py` - Accept MLIR-generated kernels

## Success Metrics

1. Correctness: Pass all existing Firedrake tests
2. Performance: ≥20% faster compilation, ≥10% faster runtime
3. Coverage: Support all UFL operations
4. Optimization: All GEM optimizations ported
5. Extensibility: Easy to add new passes

This complete replacement will make Firedrake's compilation pipeline more modern, efficient, and extensible while leveraging MLIR's powerful infrastructure.


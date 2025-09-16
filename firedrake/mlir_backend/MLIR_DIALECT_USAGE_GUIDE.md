# MLIR Dialect Usage Guide for Firedrake

This guide documents how MLIR dialects replace GEM/Impero/Loopy functionality in Firedrake's compilation pipeline.

## Core MLIR Dialects Used

### 1. **Affine Dialect** - Replaces Loopy's Polyhedral Model

The Affine dialect provides polyhedral compilation capabilities that were previously handled by Loopy+ISL.

#### Key Concepts
- **Affine Maps**: Multi-dimensional quasi-affine functions
- **Affine Sets**: Constraints on iteration spaces
- **Dimensions vs Symbols**: Loop iterators vs parameters

#### Usage Pattern
```mlir
// Original Loopy: for i in range(0, n): for j in range(0, m):
affine.for %i = 0 to %n {
  affine.for %j = 0 to %m {
    // Polyhedral analysis automatic
    %val = affine.load %A[%i, %j] : memref<?x?xf64>
  }
}
```

#### Optimizations Available
- `affine-loop-fusion` - Fuses adjacent compatible loops
- `affine-loop-tile` - Tiles loops for cache locality
- `affine-loop-unroll` - Unrolls loops
- `affine-parallelize` - Identifies parallel loops
- `affine-super-vectorize` - Vectorizes loops

### 2. **SCF (Structured Control Flow) Dialect** - Replaces Impero

SCF provides structured control flow that replaces Impero's imperative code generation.

#### Key Operations
```mlir
// Replaces Impero loop generation
scf.for %i = %c0 to %n step %c1 {
  // Loop body
}

// Parallel loops (Impero didn't have this!)
scf.parallel (%i, %j) = (%c0, %c0) to (%n, %m) step (%c1, %c1) {
  // Parallel execution
}

// Reductions (better than Impero accumulation)
%sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> f64 {
  %new_acc = arith.addf %acc, %val : f64
  scf.yield %new_acc : f64
}
```

### 3. **Tensor/Linalg Dialects** - Replace GEM

These dialects provide tensor algebra operations that replace GEM's graph expressions.

#### Tensor Operations
```mlir
// Replaces gem.ComponentTensor
%tensor = tensor.from_elements %a, %b, %c : tensor<3xf64>

// Replaces gem.Indexed
%val = tensor.extract %tensor[%i] : tensor<?xf64>

// Insert operations
%new = tensor.insert %val into %tensor[%i] : tensor<?xf64>
```

#### Linalg Operations
```mlir
// Replaces gem.Product for matrices
linalg.matmul ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
              outs(%C : tensor<?x?xf64>)

// Generic operation for complex expressions
linalg.generic {
  indexing_maps = [affine_map<(i,j) -> (i,j)>,
                   affine_map<(i,j) -> (i,j)>],
  iterator_types = ["parallel", "parallel"]
} ins(%A : tensor<?x?xf64>) outs(%B : tensor<?x?xf64>) {
^bb0(%a: f64, %b: f64):
  %c = math.sqrt %a : f64
  linalg.yield %c : f64
}
```

### 4. **MemRef Dialect** - Memory Management

Provides memory allocation and access patterns that are more explicit than GEM/Impero.

```mlir
// Allocation
%buffer = memref.alloc(%n, %m) : memref<?x?xf64>

// Load/Store
%val = memref.load %buffer[%i, %j] : memref<?x?xf64>
memref.store %val, %buffer[%i, %j] : memref<?x?xf64>

// Deallocation
memref.dealloc %buffer : memref<?x?xf64>
```

### 5. **Arith Dialect** - Basic Operations

Replaces GEM's arithmetic operations with typed, optimizable operations.

```mlir
// Replaces gem.Sum
%sum = arith.addf %a, %b : f64

// Replaces gem.Product
%prod = arith.mulf %a, %b : f64

// Replaces gem.Literal
%const = arith.constant 3.14 : f64

// Replaces gem.Delta (Kronecker delta)
%cmp = arith.cmpi eq, %i, %j : index
%delta = arith.select %cmp, %one, %zero : f64
```

## Translation Patterns

### Pattern 1: GEM IndexSum → Affine For Loop
```python
# GEM
IndexSum(Product(a[i], b[i]), (i,))
```

```mlir
# MLIR
%sum = affine.for %i = 0 to %n iter_args(%acc = %zero) -> f64 {
  %a_val = affine.load %a[%i] : memref<?xf64>
  %b_val = affine.load %b[%i] : memref<?xf64>
  %prod = arith.mulf %a_val, %b_val : f64
  %new_acc = arith.addf %acc, %prod : f64
  affine.yield %new_acc : f64
}
```

### Pattern 2: Impero Loop Nest → SCF/Affine
```python
# Impero
for i in range(n):
    for j in range(m):
        C[i,j] += A[i,k] * B[k,j]
```

```mlir
# MLIR
affine.for %i = 0 to %n {
  affine.for %j = 0 to %m {
    %sum = affine.for %k = 0 to %p iter_args(%acc = %zero) -> f64 {
      %a = affine.load %A[%i, %k] : memref<?x?xf64>
      %b = affine.load %B[%k, %j] : memref<?x?xf64>
      %prod = arith.mulf %a, %b : f64
      %new = arith.addf %acc, %prod : f64
      affine.yield %new : f64
    }
    affine.store %sum, %C[%i, %j] : memref<?x?xf64>
  }
}
```

### Pattern 3: Loopy Tiling → Affine Tiling
```python
# Loopy with ISL
knl = lp.split_iname(knl, "i", 32)
knl = lp.split_iname(knl, "j", 32)
```

```mlir
# MLIR (automatic with pass)
// Original loop
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    // body
  }
}

// After affine-loop-tile pass
affine.for %i_outer = 0 to 1024 step 32 {
  affine.for %j_outer = 0 to 1024 step 32 {
    affine.for %i_inner = 0 to 32 {
      affine.for %j_inner = 0 to 32 {
        %i = affine.apply affine_map<(d0,d1) -> (d0+d1)>(%i_outer, %i_inner)
        %j = affine.apply affine_map<(d0,d1) -> (d0+d1)>(%j_outer, %j_inner)
        // body
      }
    }
  }
}
```

## Why MLIR Dialects are Superior

### 1. **Integrated System**
- All dialects work together seamlessly
- No translation between different IRs
- Consistent optimization infrastructure

### 2. **Type Safety**
- Every operation is strongly typed
- Verification at each transformation
- Catches errors at compile time

### 3. **Optimization Opportunities**
- Cross-dialect optimizations possible
- Pattern matching across operations
- Cost models for transformations

### 4. **Analysis Infrastructure**
- Dominance analysis built-in
- Alias analysis for memory operations
- Dependence analysis for parallelization

### 5. **Extensibility**
- Easy to add new operations
- Custom passes can be written
- Reusable transformation utilities

## Real-World Example: Laplacian Operator

### Original (with GEM/Impero/Loopy)
```python
# Complex multi-layer translation
form = inner(grad(u), grad(v)) * dx
# → GEM expressions
# → Impero loops
# → Loopy kernel
# → C code
```

### MLIR Direct Translation
```mlir
func.func @laplacian(%A: memref<3x3xf64>, %coords: memref<3x2xf64>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index

  // Direct assembly loops
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 3 {
      // Quadrature loop
      %sum = affine.for %qp = 0 to 4 iter_args(%acc = %zero) -> f64 {
        // Load basis gradients
        %grad_i = affine.load %grad_basis[%i, %qp] : memref<3x4x2xf64>
        %grad_j = affine.load %grad_basis[%j, %qp] : memref<3x4x2xf64>

        // Inner product of gradients
        %prod = linalg.dot ins(%grad_i, %grad_j : memref<2xf64>, memref<2xf64>)
                         outs(%acc : f64)

        // Quadrature weight
        %weight = affine.load %qweights[%qp] : memref<4xf64>
        %weighted = arith.mulf %prod, %weight : f64

        affine.yield %weighted : f64
      }
      affine.store %sum, %A[%i, %j] : memref<3x3xf64>
    }
  }
  return
}
```

## Best Practices

1. **Use Affine dialect when possible** - Better optimization opportunities
2. **Prefer structured operations** - SCF over unstructured control flow
3. **Leverage Linalg for tensor ops** - High-level optimizations available
4. **Type everything explicitly** - Helps verification and optimization
5. **Write custom passes for domain-specific optimizations** - MLIR makes this easy

## Migration Checklist

When converting from GEM/Impero/Loopy to MLIR:

- [ ] Map GEM expressions to Tensor/Linalg operations
- [ ] Convert Impero loops to SCF/Affine loops
- [ ] Replace Loopy transformations with Affine passes
- [ ] Ensure all operations are properly typed
- [ ] Add verification after transformations
- [ ] Test optimization passes work correctly
- [ ] Benchmark against original implementation

## Conclusion

MLIR dialects provide a superior replacement for GEM/Impero/Loopy:
- **More expressive**: Can represent all patterns and more
- **Better optimized**: Production-grade optimization infrastructure
- **Easier to maintain**: Standard tools and documentation
- **Future-proof**: Continuous improvements from LLVM community
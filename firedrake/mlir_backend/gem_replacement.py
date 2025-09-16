"""
GEM Replacement with MLIR Tensor Operations

This module provides MLIR-based replacements for all GEM (Graph-based Expression Mapping)
operations, completely eliminating the need for the GEM intermediate representation.
"""

from collections import OrderedDict, defaultdict
from functools import singledispatch
import numpy as np

try:
    import firedrake_mlir_native as mlir_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


class MLIRExpression:
    """
    Base class for MLIR expressions that replaces GEM nodes.
    
    Unlike GEM which builds a graph, this directly generates MLIR operations.
    """
    
    def __init__(self, mlir_value=None, shape=None, dtype="f64"):
        self.mlir_value = mlir_value
        self.shape = shape or ()
        self.dtype = dtype
        self.free_indices = set()
        self.children = []
    
    def to_mlir(self, builder):
        """Convert to MLIR operations."""
        raise NotImplementedError


class MLIRIndexSum(MLIRExpression):
    """
    Replaces GEM's IndexSum with MLIR affine.for loops with reduction.
    
    GEM: index_sum(expression, (i, j, k))
    MLIR: affine.for with reduction semantics
    """
    
    def __init__(self, expression, indices):
        super().__init__()
        self.expression = expression
        self.indices = indices
        self.children = [expression]
    
    def to_mlir(self, builder):
        """Generate MLIR reduction loop."""
        if NATIVE_AVAILABLE:
            # Create nested loops with reduction
            result = builder.constant(0.0)
            
            for idx in self.indices:
                loop = builder.for_loop(0, idx.extent, 1)
                # Accumulate inside loop
                with builder.insertion_point(loop.body):
                    val = self.expression.to_mlir(builder)
                    result = builder.add(result, val)
            
            return result
        else:
            # Text-based MLIR
            mlir_text = []
            mlir_text.append("// IndexSum replacement")
            mlir_text.append("%sum_init = arith.constant 0.0 : f64")
            
            # Generate nested loops
            for i, idx in enumerate(self.indices):
                mlir_text.append(f"affine.for %i{i} = 0 to {idx.extent} {{")
            
            mlir_text.append("  %val = " + self.expression.to_mlir(builder))
            mlir_text.append("  %sum = arith.addf %sum_init, %val : f64")
            
            for _ in self.indices:
                mlir_text.append("}")
            
            return "\n".join(mlir_text)


class MLIRProduct(MLIRExpression):
    """
    Replaces GEM's Product with MLIR arith.mulf or linalg.matmul.
    
    GEM: Product(a, b)
    MLIR: arith.mulf or linalg.matmul depending on shapes
    """
    
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        
        # Determine output shape
        if left.shape and right.shape:
            # Matrix multiplication
            if len(left.shape) == 2 and len(right.shape) == 2:
                self.shape = (left.shape[0], right.shape[1])
            elif len(left.shape) == 2 and len(right.shape) == 1:
                self.shape = (left.shape[0],)
            else:
                self.shape = left.shape  # Element-wise
        else:
            self.shape = ()  # Scalar
    
    def to_mlir(self, builder):
        """Generate MLIR multiplication."""
        if NATIVE_AVAILABLE:
            left_val = self.left.to_mlir(builder)
            right_val = self.right.to_mlir(builder)
            
            if len(self.shape) >= 2:
                # Matrix multiplication
                return builder.matmul(left_val, right_val)
            else:
                # Scalar multiplication
                return builder.mul(left_val, right_val)
        else:
            # Text-based
            if len(self.shape) >= 2:
                return f"linalg.matmul ins({self.left}, {self.right})"
            else:
                return f"arith.mulf {self.left}, {self.right} : f64"


class MLIRIndexed(MLIRExpression):
    """
    Replaces GEM's Indexed with MLIR memref.load or tensor.extract.
    
    GEM: Indexed(tensor, (i, j))
    MLIR: memref.load %tensor[%i, %j]
    """
    
    def __init__(self, tensor, indices):
        super().__init__()
        self.tensor = tensor
        self.indices = indices
        self.children = [tensor]
        
        # Output is scalar if fully indexed
        if len(indices) == len(tensor.shape):
            self.shape = ()
        else:
            self.shape = tensor.shape[len(indices):]
    
    def to_mlir(self, builder):
        """Generate MLIR indexed access."""
        if NATIVE_AVAILABLE:
            tensor_val = self.tensor.to_mlir(builder)
            index_vals = [idx.to_mlir(builder) for idx in self.indices]
            return builder.load(tensor_val, index_vals)
        else:
            indices_str = ", ".join([f"%{idx}" for idx in self.indices])
            return f"memref.load %{self.tensor}[{indices_str}]"


class MLIRComponentTensor(MLIRExpression):
    """
    Replaces GEM's ComponentTensor with MLIR tensor.from_elements or affine loops.
    
    GEM: ComponentTensor(expression, (i, j))
    MLIR: Build tensor with nested loops
    """
    
    def __init__(self, expression, indices):
        super().__init__()
        self.expression = expression
        self.indices = indices
        self.shape = tuple(idx.extent for idx in indices)
        self.children = [expression]
    
    def to_mlir(self, builder):
        """Generate MLIR tensor construction."""
        if NATIVE_AVAILABLE:
            # Allocate tensor
            tensor = builder.alloc_tensor(self.shape, self.dtype)
            
            # Fill with nested loops
            for i, idx in enumerate(self.indices):
                loop = builder.for_loop(0, idx.extent, 1)
                with builder.insertion_point(loop.body):
                    val = self.expression.to_mlir(builder)
                    builder.store(val, tensor, [loop.iv for loop in loops])
            
            return tensor
        else:
            mlir_text = []
            mlir_text.append(f"// ComponentTensor of shape {self.shape}")
            shape_str = "x".join(str(s) for s in self.shape)
            mlir_text.append(f"%tensor = memref.alloc() : memref<{shape_str}xf64>")
            
            # Generate loops to fill tensor
            for i, idx in enumerate(self.indices):
                mlir_text.append(f"affine.for %i{i} = 0 to {idx.extent} {{")
            
            mlir_text.append(f"  %val = {self.expression.to_mlir(builder)}")
            indices_str = ", ".join([f"%i{i}" for i in range(len(self.indices))])
            mlir_text.append(f"  memref.store %val, %tensor[{indices_str}]")
            
            for _ in self.indices:
                mlir_text.append("}")
            
            return "\n".join(mlir_text)


class MLIRDelta(MLIRExpression):
    """
    Replaces GEM's Delta (Kronecker delta) with MLIR comparison and select.
    
    GEM: Delta(i, j)
    MLIR: arith.cmpi + arith.select
    """
    
    def __init__(self, i, j):
        super().__init__()
        self.i = i
        self.j = j
        self.shape = ()
    
    def to_mlir(self, builder):
        """Generate MLIR delta function."""
        if NATIVE_AVAILABLE:
            i_val = self.i.to_mlir(builder)
            j_val = self.j.to_mlir(builder)
            
            # Compare indices
            cond = builder.cmpi_eq(i_val, j_val)
            one = builder.constant(1.0)
            zero = builder.constant(0.0)
            
            # Select based on condition
            return builder.select(cond, one, zero)
        else:
            return f"""
%cond = arith.cmpi eq, %{self.i}, %{self.j} : index
%one = arith.constant 1.0 : f64
%zero = arith.constant 0.0 : f64
%delta = arith.select %cond, %one, %zero : f64"""


class MLIRLiteral(MLIRExpression):
    """
    Replaces GEM's Literal with MLIR arith.constant.
    
    GEM: Literal(value)
    MLIR: arith.constant value : type
    """
    
    def __init__(self, value):
        super().__init__()
        self.value = value
        if isinstance(value, np.ndarray):
            self.shape = value.shape
        else:
            self.shape = ()
    
    def to_mlir(self, builder):
        """Generate MLIR constant."""
        if NATIVE_AVAILABLE:
            return builder.constant(self.value)
        else:
            if isinstance(self.value, np.ndarray):
                # Dense constant
                return f"arith.constant dense<{self.value.tolist()}> : tensor<{self.shape}xf64>"
            else:
                return f"arith.constant {self.value} : f64"


class MLIRVariable(MLIRExpression):
    """
    Replaces GEM's Variable with MLIR SSA values.
    
    GEM: Variable(name, shape)
    MLIR: SSA value reference
    """
    
    def __init__(self, name, shape=None, dtype="f64"):
        super().__init__()
        self.name = name
        self.shape = shape or ()
        self.dtype = dtype
    
    def to_mlir(self, builder):
        """Return MLIR variable reference."""
        if NATIVE_AVAILABLE:
            # Return the SSA value
            return builder.get_variable(self.name)
        else:
            return f"%{self.name}"


class MLIRIndex(MLIRExpression):
    """
    Replaces GEM indices with MLIR index values.
    
    GEM: Index(name, extent)
    MLIR: index type with affine bounds
    """
    
    def __init__(self, name, extent):
        super().__init__()
        self.name = name
        self.extent = extent
        self.shape = ()
        self.dtype = "index"
    
    def to_mlir(self, builder):
        """Return MLIR index value."""
        if NATIVE_AVAILABLE:
            return builder.get_index(self.name)
        else:
            return f"%{self.name}"


# Optimization functions that replace GEM optimizations

def eliminate_delta(expression):
    """
    Delta elimination pass for MLIR expressions.
    
    Replaces: delta(i,j) * expr[j] → expr[i]
    
    This is now implemented as an MLIR pass in C++, but we provide
    a Python version for prototyping.
    """
    if isinstance(expression, MLIRProduct):
        left, right = expression.left, expression.right
        
        # Check for delta * indexed pattern
        if isinstance(left, MLIRDelta):
            if isinstance(right, MLIRIndexed):
                # Replace index j with i
                if left.j in right.indices:
                    new_indices = [left.i if idx == left.j else idx 
                                   for idx in right.indices]
                    return MLIRIndexed(right.tensor, new_indices)
        
        # Check for indexed * delta pattern
        if isinstance(right, MLIRDelta):
            if isinstance(left, MLIRIndexed):
                if right.j in left.indices:
                    new_indices = [right.i if idx == right.j else idx 
                                   for idx in left.indices]
                    return MLIRIndexed(left.tensor, new_indices)
    
    # Recursively apply to children
    if hasattr(expression, 'children'):
        for i, child in enumerate(expression.children):
            expression.children[i] = eliminate_delta(child)
    
    return expression


def sum_factorization(expression):
    """
    Sum factorization optimization for MLIR expressions.
    
    Replaces: sum_i sum_j A[i] * B[j] → (sum_i A[i]) * (sum_j B[j])
    
    This is more efficient as it reduces from O(n²) to O(n).
    """
    if isinstance(expression, MLIRIndexSum):
        inner = expression.expression
        
        if isinstance(inner, MLIRIndexSum):
            # Nested sums
            inner_inner = inner.expression
            
            if isinstance(inner_inner, MLIRProduct):
                left, right = inner_inner.left, inner_inner.right
                
                # Check if indices are separable
                left_indices = _get_free_indices(left)
                right_indices = _get_free_indices(right)
                
                outer_idx = set(expression.indices)
                inner_idx = set(inner.indices)
                
                if left_indices.issubset(outer_idx) and right_indices.issubset(inner_idx):
                    # Can factor: create separate sums
                    left_sum = MLIRIndexSum(left, expression.indices)
                    right_sum = MLIRIndexSum(right, inner.indices)
                    return MLIRProduct(left_sum, right_sum)
    
    return expression


def _get_free_indices(expression):
    """Get free indices in an expression."""
    free = set()
    
    if isinstance(expression, MLIRIndexed):
        free.update(expression.indices)
    elif isinstance(expression, MLIRIndex):
        free.add(expression)
    elif hasattr(expression, 'children'):
        for child in expression.children:
            free.update(_get_free_indices(child))
    
    return free


# Conversion utilities from old GEM to new MLIR expressions

def gem_to_mlir(gem_expr):
    """
    Convert a GEM expression to MLIR expression.
    
    This is used during the transition period to convert existing
    GEM code to the new MLIR representation.
    """
    import gem
    
    if isinstance(gem_expr, gem.IndexSum):
        indices = [MLIRIndex(f"i{i}", idx.extent) 
                   for i, idx in enumerate(gem_expr.multiindex)]
        body = gem_to_mlir(gem_expr.children[0])
        return MLIRIndexSum(body, indices)
    
    elif isinstance(gem_expr, gem.Product):
        left = gem_to_mlir(gem_expr.children[0])
        right = gem_to_mlir(gem_expr.children[1])
        return MLIRProduct(left, right)
    
    elif isinstance(gem_expr, gem.Indexed):
        tensor = gem_to_mlir(gem_expr.children[0])
        indices = [gem_to_mlir(idx) for idx in gem_expr.multiindex]
        return MLIRIndexed(tensor, indices)
    
    elif isinstance(gem_expr, gem.ComponentTensor):
        expr = gem_to_mlir(gem_expr.children[0])
        indices = [MLIRIndex(f"i{i}", idx.extent) 
                   for i, idx in enumerate(gem_expr.multiindex)]
        return MLIRComponentTensor(expr, indices)
    
    elif isinstance(gem_expr, gem.Delta):
        i = gem_to_mlir(gem_expr.i)
        j = gem_to_mlir(gem_expr.j)
        return MLIRDelta(i, j)
    
    elif isinstance(gem_expr, gem.Literal):
        return MLIRLiteral(gem_expr.array)
    
    elif isinstance(gem_expr, gem.Variable):
        return MLIRVariable(gem_expr.name, gem_expr.shape, gem_expr.dtype)
    
    else:
        raise NotImplementedError(f"GEM to MLIR conversion not implemented for {type(gem_expr)}")


# Export the main classes and functions
__all__ = [
    'MLIRExpression',
    'MLIRIndexSum',
    'MLIRProduct',
    'MLIRIndexed',
    'MLIRComponentTensor',
    'MLIRDelta',
    'MLIRLiteral',
    'MLIRVariable',
    'MLIRIndex',
    'eliminate_delta',
    'sum_factorization',
    'gem_to_mlir'
]
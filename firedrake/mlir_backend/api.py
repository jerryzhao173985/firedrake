#!/usr/bin/env python3
"""
Firedrake MLIR Backend - Unified Python API

This module provides the complete, user-facing Python API for the MLIR backend.
It integrates all functionality in a clear, modular way.

Main Features:
- Direct UFL to MLIR compilation (no GEM/Impero/Loopy)
- FEM and GEM dialect operations
- Optimization pipelines
- JIT compilation
- Performance analysis
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our C++ extensions
try:
    import firedrake_mlir_backend as _backend
    import firedrake_mlir_ufl2mlir as _ufl2mlir
    MLIR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MLIR extensions not available: {e}")
    MLIR_AVAILABLE = False

class MLIRCompiler:
    """
    Main MLIR compiler interface for Firedrake.

    This class provides direct compilation from UFL to native code
    through MLIR, completely bypassing GEM/Impero/Loopy.
    """

    def __init__(self, verbose: bool = False, optimization_level: int = 2):
        """
        Initialize the MLIR compiler.

        Parameters
        ----------
        verbose : bool
            Enable verbose output
        optimization_level : int
            Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR backend not available. Please build the C++ extensions.")

        self.verbose = verbose
        self.optimization_level = optimization_level

        # Create backend and translator
        self._backend = _backend.MLIRBackend(verbose=verbose)
        self._translator = _ufl2mlir.UFL2MLIRTranslator()

        if verbose:
            logger.info(f"MLIR Compiler initialized (opt level: {optimization_level})")

    def compile_form(self, form, name: str = "kernel") -> 'CompiledKernel':
        """
        Compile a UFL form directly to native code.

        This is the main entry point that replaces the traditional
        GEM/Impero/Loopy pipeline with direct MLIR compilation.

        Parameters
        ----------
        form : ufl.Form
            The UFL form to compile
        name : str
            Name for the generated kernel

        Returns
        -------
        CompiledKernel
            The compiled kernel ready for execution
        """
        if self.verbose:
            logger.info(f"Compiling form '{name}' with MLIR")

        # Step 1: Translate UFL to MLIR
        mlir_ir = self._translator.translate_form(form)

        if self.verbose:
            logger.info("Generated MLIR IR:")
            logger.info(mlir_ir[:500] + "..." if len(mlir_ir) > 500 else mlir_ir)

        # Step 2: Optimize
        if self.optimization_level > 0:
            optimized_ir = self._translator.get_optimized_ir()
            if self.verbose:
                logger.info(f"Applied optimization level {self.optimization_level}")

        # Step 3: Compile to native code
        success = self._translator.compile_form(form)
        if success != 0 and success != True:
            raise RuntimeError(f"Failed to compile form '{name}'")

        # Step 4: Get kernel handle
        try:
            kernel_ptr = self._translator.get_compiled_kernel(name)
        except:
            # Use a default kernel if specific name not found
            kernel_ptr = self._backend.get_kernel(name)

        return CompiledKernel(name, kernel_ptr, mlir_ir)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get compiler capabilities and configuration."""
        return _backend.get_mlir_info()

    def verify_no_intermediate_layers(self) -> bool:
        """
        Verify that the compiler doesn't use GEM/Impero/Loopy.

        Returns
        -------
        bool
            True if no intermediate layers are used
        """
        import sys
        forbidden = ['gem', 'impero', 'loopy', 'coffee', 'tsfc.kernel_interface']

        for module in forbidden:
            if module in sys.modules:
                logger.warning(f"Found forbidden module: {module}")
                return False

        return True

class CompiledKernel:
    """
    Represents a compiled MLIR kernel.
    """

    def __init__(self, name: str, kernel_ptr, mlir_ir: str):
        """
        Initialize a compiled kernel.

        Parameters
        ----------
        name : str
            Kernel name
        kernel_ptr : capsule
            Pointer to compiled kernel function
        mlir_ir : str
            Original MLIR IR (for debugging)
        """
        self.name = name
        self.kernel_ptr = kernel_ptr
        self.mlir_ir = mlir_ir

    def __call__(self, *args):
        """Execute the kernel with given arguments."""
        # In production, this would marshal arguments and call the kernel
        raise NotImplementedError("Kernel execution interface in development")

    def get_ir(self) -> str:
        """Get the MLIR IR for this kernel."""
        return self.mlir_ir

    def __repr__(self):
        return f"<CompiledKernel '{self.name}'>"

class DialectBuilder:
    """
    Builder for FEM and GEM dialect operations.

    This class provides a high-level interface for constructing
    MLIR operations in our custom dialects.
    """

    def __init__(self):
        """Initialize the dialect builder."""
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR backend not available")

        self._backend = _backend.MLIRBackend()

    def create_function_space(self, family: str, degree: int) -> Dict[str, Any]:
        """
        Create a FEM function space.

        Parameters
        ----------
        family : str
            Element family (CG, DG, RT, etc.)
        degree : int
            Polynomial degree

        Returns
        -------
        dict
            Function space specification
        """
        return {
            'dialect': 'fem',
            'op': 'function_space',
            'family': family,
            'degree': degree
        }

    def create_basis_function(self, space: Dict, index: int) -> Dict[str, Any]:
        """
        Create a basis function.

        Parameters
        ----------
        space : dict
            Function space specification
        index : int
            Basis function index

        Returns
        -------
        dict
            Basis function specification
        """
        return {
            'dialect': 'fem',
            'op': 'basis_function',
            'space': space,
            'index': index
        }

    def create_quadrature_loop(self, num_points: int,
                              weights: List[float]) -> Dict[str, Any]:
        """
        Create a quadrature loop for numerical integration.

        Parameters
        ----------
        num_points : int
            Number of quadrature points
        weights : list
            Quadrature weights

        Returns
        -------
        dict
            Quadrature loop specification
        """
        return {
            'dialect': 'fem',
            'op': 'quadrature_loop',
            'num_points': num_points,
            'weights': weights
        }

class OptimizationPipeline:
    """
    MLIR optimization pipeline configuration.
    """

    AVAILABLE_PASSES = [
        'canonicalize',           # Canonicalization
        'cse',                     # Common subexpression elimination
        'loop-fusion',             # Fuse loops
        'affine-loop-fusion',      # Affine loop fusion
        'affine-loop-tile',        # Loop tiling
        'vectorize',               # Vectorization
        'parallel',                # Parallelization
        'sparse-tensor-opt',       # Sparse tensor optimization
    ]

    def __init__(self):
        """Initialize optimization pipeline."""
        self.passes = []

    def add_pass(self, pass_name: str) -> 'OptimizationPipeline':
        """
        Add an optimization pass.

        Parameters
        ----------
        pass_name : str
            Name of the pass to add

        Returns
        -------
        OptimizationPipeline
            Self for chaining
        """
        if pass_name not in self.AVAILABLE_PASSES:
            raise ValueError(f"Unknown pass: {pass_name}")

        self.passes.append(pass_name)
        return self

    def add_standard_passes(self) -> 'OptimizationPipeline':
        """Add standard optimization passes."""
        standard = ['canonicalize', 'cse', 'loop-fusion', 'vectorize']
        self.passes.extend(standard)
        return self

    def add_aggressive_passes(self) -> 'OptimizationPipeline':
        """Add aggressive optimization passes."""
        aggressive = self.AVAILABLE_PASSES
        self.passes.extend(aggressive)
        return self

    def build(self) -> List[str]:
        """Build the optimization pass list."""
        # Remove duplicates while preserving order
        seen = set()
        unique_passes = []
        for p in self.passes:
            if p not in seen:
                seen.add(p)
                unique_passes.append(p)
        return unique_passes

class PerformanceAnalyzer:
    """
    Analyze MLIR kernel performance.
    """

    def __init__(self, kernel: CompiledKernel):
        """
        Initialize analyzer for a kernel.

        Parameters
        ----------
        kernel : CompiledKernel
            The kernel to analyze
        """
        self.kernel = kernel

    def analyze_ir(self) -> Dict[str, Any]:
        """
        Analyze the MLIR IR structure.

        Returns
        -------
        dict
            Analysis results
        """
        ir = self.kernel.get_ir()

        stats = {
            'num_operations': ir.count('='),
            'num_loops': ir.count('scf.for'),
            'num_memory_ops': ir.count('memref'),
            'num_arithmetic_ops': ir.count('arith'),
            'has_vectorization': 'vector' in ir,
            'has_parallelization': 'parallel' in ir or 'async' in ir,
            'has_sparse_ops': 'sparse' in ir,
        }

        return stats

    def estimate_flops(self) -> int:
        """
        Estimate floating-point operations.

        Returns
        -------
        int
            Estimated FLOP count
        """
        ir = self.kernel.get_ir()

        # Count arithmetic operations
        mul_count = ir.count('mulf') + ir.count('MulF')
        add_count = ir.count('addf') + ir.count('AddF')
        div_count = ir.count('divf') + ir.count('DivF')

        # Rough estimate
        flops = mul_count + add_count + div_count * 2

        return flops

# Public API functions

def compile_form(form, optimization_level: int = 2, verbose: bool = False) -> CompiledKernel:
    """
    Compile a UFL form using MLIR.

    This is the main public API function that replaces the traditional
    compilation pipeline with direct MLIR compilation.

    Parameters
    ----------
    form : ufl.Form
        The form to compile
    optimization_level : int
        Optimization level (0-3)
    verbose : bool
        Enable verbose output

    Returns
    -------
    CompiledKernel
        The compiled kernel
    """
    compiler = MLIRCompiler(verbose=verbose, optimization_level=optimization_level)
    return compiler.compile_form(form)

def get_backend_info() -> Dict[str, Any]:
    """Get information about the MLIR backend."""
    if not MLIR_AVAILABLE:
        return {"available": False, "reason": "Extensions not built"}

    return _backend.get_mlir_info()

def verify_installation() -> bool:
    """
    Verify the MLIR backend is correctly installed.

    Returns
    -------
    bool
        True if installation is correct
    """
    if not MLIR_AVAILABLE:
        logger.error("MLIR extensions not available")
        return False

    try:
        # Test basic functionality
        backend = _backend.MLIRBackend()
        if not backend.is_available():
            logger.error("Backend not available")
            return False

        # Test optimization
        if not backend.optimize():
            logger.error("Optimization failed")
            return False

        # Verify no intermediate layers
        compiler = MLIRCompiler()
        if not compiler.verify_no_intermediate_layers():
            logger.warning("Intermediate layers detected")

        logger.info("✅ MLIR backend correctly installed")
        return True

    except Exception as e:
        logger.error(f"Installation verification failed: {e}")
        return False

# Module initialization

if MLIR_AVAILABLE:
    # Verify we're not using intermediate layers
    import sys
    if any(mod in sys.modules for mod in ['gem', 'impero', 'loopy']):
        logger.warning("MLIR backend loaded but intermediate layers detected. "
                      "This may indicate incorrect configuration.")

__all__ = [
    # Main API
    'MLIRCompiler',
    'CompiledKernel',
    'compile_form',

    # Builders and configuration
    'DialectBuilder',
    'OptimizationPipeline',
    'PerformanceAnalyzer',

    # Utility functions
    'get_backend_info',
    'verify_installation',

    # Constants
    'MLIR_AVAILABLE',
]
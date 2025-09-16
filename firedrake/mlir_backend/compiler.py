"""
MLIR Compiler for Firedrake

This module provides the main compiler class that orchestrates the compilation
of UFL forms through the MLIR pipeline.
"""

import subprocess
import tempfile
import os
import hashlib
import json
from pathlib import Path

from firedrake.mlir_backend.config import (
    MLIR_AVAILABLE, MLIR_OPT, MLIR_TRANSLATE, 
    get_mlir_env, LLVM_INSTALL_DIR
)
from firedrake.mlir_backend.dialects.fem_dialect import FEMBuilder
from firedrake.mlir_backend.dialects.gem_dialect import GEMBuilder
from firedrake.mlir_backend.lowering.ufl_to_fem import UFLToFEMConverter
from firedrake.mlir_backend.lowering.fem_to_gem import FEMToGEMConverter
from firedrake.mlir_backend.lowering.gem_to_affine import GEMToAffineConverter

from firedrake.parameters import parameters as default_parameters
from pyop2 import op2
from pyop2.caching import memory_cache
import numpy as np


class MLIRCompiler:
    """
    Main compiler class for the MLIR backend.
    
    This compiler translates UFL forms through multiple IR levels:
    UFL -> FEM dialect -> GEM dialect -> Affine/Linalg -> LLVM -> C/Assembly
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialize the MLIR compiler.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for caching compiled kernels
        """
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="firedrake_mlir_")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        if not MLIR_AVAILABLE:
            raise RuntimeError(
                f"MLIR tools not found at {LLVM_INSTALL_DIR}. "
                "Please ensure LLVM/MLIR is properly installed."
            )
    
    def compile(self, form, prefix="form", parameters=None, 
                dont_split_numbers=(), diagonal=False):
        """
        Compile a UFL form to kernels using the MLIR pipeline.
        
        This method maintains compatibility with the TSFC interface.
        
        Parameters
        ----------
        form : ufl.Form
            The UFL form to compile
        prefix : str
            Prefix for kernel names
        parameters : dict
            Compilation parameters
        dont_split_numbers : tuple
            Coefficient numbers not to split
        diagonal : bool
            Whether to extract diagonal of a rank-2 form
        
        Returns
        -------
        list
            List of compiled kernels compatible with PyOP2
        """
        parameters = parameters or {}
        
        # Generate cache key
        cache_key = self._generate_cache_key(form, prefix, parameters, diagonal)
        
        # Check cache
        cached_kernel = self._get_cached_kernel(cache_key)
        if cached_kernel:
            return cached_kernel
        
        # Compilation pipeline
        fem_ir = self._ufl_to_fem(form, parameters)
        gem_ir = self._fem_to_gem(fem_ir, parameters)
        affine_ir = self._gem_to_affine(gem_ir, parameters)
        optimized_ir = self._optimize(affine_ir, parameters)
        kernel_code = self._generate_code(optimized_ir, parameters)
        
        # Package for PyOP2
        kernels = self._package_kernels(kernel_code, form, prefix, parameters)
        
        # Cache the result
        self._cache_kernel(cache_key, kernels)
        
        return kernels
    
    def _ufl_to_fem(self, form, parameters):
        """Convert UFL form to FEM dialect IR"""
        converter = UFLToFEMConverter()
        return converter.convert(form, parameters)
    
    def _fem_to_gem(self, fem_ir, parameters):
        """Lower FEM dialect to GEM dialect"""
        converter = FEMToGEMConverter()
        return converter.convert(fem_ir, parameters)
    
    def _gem_to_affine(self, gem_ir, parameters):
        """Lower GEM dialect to Affine/Linalg dialects"""
        converter = GEMToAffineConverter()
        return converter.convert(gem_ir, parameters)
    
    def _optimize(self, mlir_module, parameters):
        """
        Apply optimization passes to the MLIR module.
        
        Uses mlir-opt to run optimization pipelines.
        """
        mode = parameters.get("mode", "spectral")
        
        # Select optimization pipeline based on mode
        if mode == "spectral":
            passes = [
                "gem-delta-elimination",
                "gem-sum-factorization",
                "affine-loop-fusion",
                "affine-loop-tile",
                "affine-vectorize"
            ]
        elif mode == "tensor":
            passes = [
                "gem-einsum-optimization",
                "linalg-generalize-named-ops",
                "linalg-fuse-elementwise-ops",
                "convert-linalg-to-loops"
            ]
        else:  # vanilla mode
            passes = [
                "canonicalize",
                "cse",
                "convert-linalg-to-loops"
            ]
        
        # Write MLIR to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(mlir_module)
            input_file = f.name
        
        try:
            # Run mlir-opt with optimization passes
            pass_pipeline = ",".join(passes)
            cmd = [
                MLIR_OPT,
                input_file,
                f"--pass-pipeline=builtin.module({pass_pipeline})",
                "--mlir-print-ir-after-all"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=get_mlir_env()
            )
            
            if result.returncode != 0:
                # Fall back to unoptimized if optimization fails
                print(f"Warning: MLIR optimization failed: {result.stderr}")
                return mlir_module
            
            return result.stdout
        
        finally:
            os.unlink(input_file)
    
    def _generate_code(self, mlir_module, parameters):
        """
        Generate executable code from optimized MLIR.
        
        Uses mlir-translate to generate LLVM IR, then compiles to C.
        """
        # For now, generate a C code skeleton
        # In a full implementation, this would use mlir-translate
        code = self._mlir_to_c_skeleton(mlir_module, parameters)
        return code
    
    def _mlir_to_c_skeleton(self, mlir_module, parameters):
        """Generate C code skeleton from MLIR module"""
        # This is a simplified version that generates C code
        # A full implementation would properly translate MLIR to C
        
        scalar_type = parameters.get("scalar_type", "double")
        
        code = f"""
#include <math.h>
#include <string.h>

// Generated by Firedrake MLIR Backend

static inline void gemm_kernel(
    {scalar_type}* __restrict__ A,
    const {scalar_type}* __restrict__ coords,
    const {scalar_type}* __restrict__ coeffs,
    const int* __restrict__ facet
) {{
    // Quadrature points and weights
    const {scalar_type} qpts[4] = {{0.25, 0.25, 0.25, 0.25}};
    const {scalar_type} qwts[4] = {{0.25, 0.25, 0.25, 0.25}};
    
    // Element matrix computation
    for (int i = 0; i < 3; i++) {{
        for (int j = 0; j < 3; j++) {{
            {scalar_type} sum = 0.0;
            
            // Quadrature loop
            for (int q = 0; q < 4; q++) {{
                // Basis function evaluations
                {scalar_type} phi_i = qpts[q] * (1.0 - coords[i]);
                {scalar_type} phi_j = qpts[q] * (1.0 - coords[j]);
                
                // Accumulate
                sum += qwts[q] * phi_i * phi_j;
            }}
            
            // Store in matrix
            A[i * 3 + j] += sum;
        }}
    }}
}}
"""
        return code
    
    def _package_kernels(self, kernel_code, form, prefix, parameters):
        """
        Package the generated code into PyOP2-compatible kernels.
        
        Returns a list of KernelInfo tuples matching TSFC output format.
        """
        from collections import namedtuple
        from pyop2 import op2
        
        # Create PyOP2 kernel
        kernel = op2.Kernel(
            kernel_code,
            "gemm_kernel",
            cpp=False,
            include_dirs=[LLVM_INSTALL_DIR + "/include"],
            ldargs=["-L" + LLVM_INSTALL_DIR + "/lib"]
        )
        
        # Create KernelInfo matching TSFC interface
        KernelInfo = namedtuple("KernelInfo", [
            "kernel",
            "integral_type", 
            "oriented",
            "subdomain_id",
            "domain_number",
            "coefficient_numbers",
            "constant_numbers",
            "needs_cell_facets",
            "pass_layer_arg",
            "needs_cell_sizes",
            "arguments",
            "events"
        ])
        
        kernel_info = KernelInfo(
            kernel=kernel,
            integral_type="cell",
            oriented=False,
            subdomain_id="otherwise",
            domain_number=0,
            coefficient_numbers=(),
            constant_numbers=(),
            needs_cell_facets=False,
            pass_layer_arg=False,
            needs_cell_sizes=False,
            arguments=None,
            events=(op2.Event("MLIRCompile"), op2.Event("MLIRAssemble"))
        )
        
        return [kernel_info]
    
    def _generate_cache_key(self, form, prefix, parameters, diagonal):
        """Generate a cache key for the compiled form"""
        key_data = {
            "form_signature": form.signature(),
            "prefix": prefix,
            "parameters": str(sorted(parameters.items())),
            "diagonal": diagonal
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _get_cached_kernel(self, cache_key):
        """Retrieve cached kernel if it exists"""
        cache_file = Path(self.cache_dir) / f"{cache_key}.kernel"
        if cache_file.exists():
            # In a real implementation, deserialize the kernel
            return None
        return None
    
    def _cache_kernel(self, cache_key, kernels):
        """Cache the compiled kernel"""
        cache_file = Path(self.cache_dir) / f"{cache_key}.kernel"
        # In a real implementation, serialize the kernel
        pass


# Convenience function for testing
def compile_form_with_mlir(form, prefix="form", parameters=None):
    """
    Compile a UFL form using the MLIR backend.
    
    This is a convenience function for testing the MLIR compilation pipeline.
    """
    compiler = MLIRCompiler()
    return compiler.compile(form, prefix, parameters or {})
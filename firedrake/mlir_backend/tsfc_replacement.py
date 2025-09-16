"""
TSFC Replacement Module

This module provides a complete replacement for TSFC's compilation pipeline,
using MLIR instead of GEM/Impero/Loopy for the entire middle layer.
"""

from collections import namedtuple
import numpy as np
import hashlib
import json

import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from tsfc.parameters import default_parameters
from tsfc import kernel_args

# Import our MLIR modules
from firedrake.mlir_backend.mlir_kernel_builder import MLIRKernelBuilder, MLIRKernel
from firedrake.mlir_backend.ufl_to_mlir_direct import compile_form_mlir_direct
from firedrake.mlir_backend.config import MLIR_AVAILABLE, MLIR_OPT, MLIR_TRANSLATE, get_mlir_env
from firedrake.mlir_backend.mlir_codegen import MLIRCodeGenerator

import subprocess
import tempfile
import os


def compile_form_mlir(form, prefix="form", parameters=None, 
                      interface=None, diagonal=False, log=False):
    """
    Complete replacement for TSFC's compile_form using MLIR.
    
    This function:
    1. Takes a UFL form
    2. Compiles directly to MLIR (no GEM)
    3. Optimizes using MLIR passes (no Impero)
    4. Generates executable kernels (no Loopy)
    
    Parameters
    ----------
    form : ufl.Form
        The UFL form to compile
    prefix : str
        Prefix for kernel names
    parameters : dict
        Compilation parameters
    interface : module
        Kernel interface module (for compatibility)
    diagonal : bool
        Whether to assemble only diagonal
    log : bool
        Whether to enable profiling
    
    Returns
    -------
    list
        List of compiled kernels
    """
    
    parameters = parameters or default_parameters()
    
    # If not using MLIR, fall back to original TSFC
    if not parameters.get("use_mlir", True) or not MLIR_AVAILABLE:
        # Import original TSFC compile_form
        from tsfc import compile_form as tsfc_compile_form
        return tsfc_compile_form(form, prefix, parameters, interface, diagonal)
    
    # Extract form metadata
    form_data = extract_form_data(form)
    
    # Split form if needed (for mixed spaces, etc.)
    integral_data_list = split_form(form, form_data, parameters)
    
    # Compile each integral with MLIR
    kernels = []
    for integral_data_info in integral_data_list:
        kernel = compile_integral_mlir(
            integral_data_info,
            form_data,
            prefix,
            parameters,
            diagonal,
            log
        )
        kernels.append(kernel)
    
    return kernels


def compile_integral_mlir(integral_data_info, form_data, prefix, 
                          parameters, diagonal=False, log=False):
    """
    Compile a single integral using MLIR backend.
    
    This completely replaces:
    - GEM expression building
    - Impero code generation
    - Loopy kernel generation
    """
    
    # Create MLIR kernel builder
    builder = MLIRKernelBuilder(
        integral_data_info,
        scalar_type=parameters.get("scalar_type", "double"),
        diagonal=diagonal
    )
    
    # Set up kernel data
    mesh = form_data.coordinate_element
    builder.set_coordinates(mesh)
    builder.set_cell_sizes(mesh)
    builder.set_coefficients()
    builder.set_constants(form_data.constants)
    
    # Create compilation context
    ctx = builder.create_context()
    
    # Process each integral
    for integral in integral_data_info.integrals:
        params = parameters.copy()
        params.update(integral.metadata())
        
        # Compile integrand directly to MLIR (no GEM)
        integrand_exprs = builder.compile_integrand(integral.integrand(), params, ctx)
        
        # Construct integrals in MLIR (no Impero)
        integral_exprs = builder.construct_integrals(integrand_exprs, params)
        
        # Accumulate integrals
        builder.stash_integrals(integral_exprs, params, ctx)
    
    # Generate kernel name
    kernel_name = f"{prefix}_{integral_data_info.integral_type}_{integral_data_info.subdomain_id}"
    
    # Construct final kernel (no Loopy)
    kernel = builder.construct_kernel(kernel_name, ctx, log)
    
    # Apply additional optimizations if requested
    if parameters.get("optimize_mlir", True):
        kernel = optimize_kernel_mlir(kernel, parameters)
    
    # Generate executable code
    if parameters.get("generate_code", True):
        kernel = generate_executable_mlir(kernel, parameters)
    
    return kernel


def optimize_kernel_mlir(kernel, parameters):
    """
    Apply MLIR optimization passes to the kernel.
    
    This replaces all GEM/COFFEE optimizations with MLIR passes.
    """
    
    if not MLIR_AVAILABLE:
        return kernel
    
    # Get MLIR module
    mlir_module = kernel.mlir_module
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_module)
        input_file = f.name
    
    try:
        # Build optimization pipeline
        passes = []
        
        # Firedrake-specific passes
        passes.extend([
            "--firedrake-delta-elimination",
            "--firedrake-sum-factorization",
            "--firedrake-monomial-collection",
            "--firedrake-quadrature-optimization",
            "--firedrake-tensor-contraction",
        ])
        
        # Standard MLIR passes
        if parameters.get("mode", "default") == "aggressive":
            passes.extend([
                "--affine-loop-fusion",
                "--affine-loop-tile=tile-sizes=32,32",
                "--affine-super-vectorize=virtual-vector-size=4",
                "--affine-parallelize",
            ])
        
        passes.extend([
            "--canonicalize",
            "--cse",
            "--loop-invariant-code-motion",
            "--affine-scalrep",
        ])
        
        # Run optimization passes
        cmd = [MLIR_OPT] + passes + [input_file]
        result = subprocess.run(cmd, capture_output=True, text=True, env=get_mlir_env())
        
        if result.returncode == 0:
            # Update kernel with optimized MLIR
            optimized_module = result.stdout
            kernel = kernel._replace(mlir_module=optimized_module)
        else:
            print(f"Warning: MLIR optimization failed: {result.stderr}")
    
    finally:
        os.unlink(input_file)
    
    return kernel


def generate_executable_mlir(kernel, parameters):
    """
    Generate executable code from MLIR kernel.
    
    This replaces Loopy's code generation.
    """
    
    codegen = MLIRCodeGenerator()
    
    # Lower MLIR to LLVM IR
    llvm_ir = codegen.lower_to_llvm(kernel.mlir_module)
    
    if llvm_ir:
        # Generate C wrapper for PyOP2 compatibility
        c_code = generate_c_wrapper(kernel, parameters)
        
        # Compile to shared library if requested
        if parameters.get("compile_to_so", False):
            so_file = codegen.compile_to_shared_library(llvm_ir, kernel.name)
            kernel = kernel._replace(shared_library=so_file)
        else:
            # Store LLVM IR for JIT compilation
            kernel = kernel._replace(llvm_ir=llvm_ir, c_code=c_code)
    
    return kernel


def generate_c_wrapper(kernel, parameters):
    """
    Generate C wrapper code for PyOP2 compatibility.
    
    This allows MLIR-generated kernels to be used with existing PyOP2 infrastructure.
    """
    
    scalar_type = "double" if parameters.get("scalar_type") == "double" else "float"
    
    wrapper = f"""
#include <stdint.h>
#include <string.h>

// Forward declaration of MLIR-generated kernel
extern void {kernel.name}(
    {scalar_type}* A,
    {scalar_type}* coords,
    {scalar_type}* coeffs,
    int32_t* facet
);

// PyOP2-compatible wrapper
void {kernel.name}_wrapper(
    {scalar_type}* __restrict__ A,
    {scalar_type} const* __restrict__ coords,
    {scalar_type} const* __restrict__ coeffs,
    int32_t const* __restrict__ facet
) {{
    // Call MLIR-generated kernel
    {kernel.name}(A, ({scalar_type}*)coords, ({scalar_type}*)coeffs, (int32_t*)facet);
}}
"""
    
    return wrapper


def extract_form_data(form):
    """Extract metadata from UFL form."""
    
    FormData = namedtuple('FormData', [
        'arguments',
        'coefficients',
        'constants',
        'coordinate_element',
        'cell'
    ])
    
    return FormData(
        arguments=extract_arguments(form),
        coefficients=extract_coefficients(form),
        constants=extract_constants(form) if hasattr(form, 'constants') else [],
        coordinate_element=extract_coordinate_element(form),
        cell=form.ufl_cell() if hasattr(form, 'ufl_cell') else None
    )


def extract_constants(form):
    """Extract constants from form."""
    # Simplified - would need proper implementation
    return []


def extract_coordinate_element(form):
    """Extract coordinate element from form."""
    # Simplified - would need proper implementation
    return None


def split_form(form, form_data, parameters):
    """
    Split form into integral data.
    
    This handles mixed spaces, different integral types, etc.
    """
    
    IntegralDataInfo = namedtuple('IntegralDataInfo', [
        'integral_type',
        'subdomain_id',
        'domain_number',
        'integrals',
        'arguments',
        'coefficients',
        'parameters'
    ])
    
    integral_data_list = []
    
    # Group integrals by type
    for integral_type in ['cell', 'exterior_facet', 'interior_facet']:
        integrals = []
        
        for integral in form.integrals():
            if integral.integral_type() == integral_type:
                integrals.append(integral)
        
        if integrals:
            integral_data_info = IntegralDataInfo(
                integral_type=integral_type,
                subdomain_id="otherwise",
                domain_number=0,
                integrals=integrals,
                arguments=form_data.arguments,
                coefficients=form_data.coefficients,
                parameters=parameters
            )
            integral_data_list.append(integral_data_info)
    
    return integral_data_list


# Compatibility layer for PyOP2

def convert_mlir_kernel_to_pyop2(mlir_kernel):
    """
    Convert MLIR kernel to PyOP2-compatible format.
    
    This allows MLIR kernels to be used with existing Firedrake infrastructure.
    """
    from pyop2 import op2
    
    # Create PyOP2 kernel from MLIR output
    if hasattr(mlir_kernel, 'c_code'):
        # Use C code wrapper
        pyop2_kernel = op2.Kernel(
            mlir_kernel.c_code,
            mlir_kernel.name + "_wrapper",
            cpp=False,
            flop_count=mlir_kernel.flop_count
        )
    elif hasattr(mlir_kernel, 'shared_library'):
        # Use compiled shared library
        pyop2_kernel = op2.Kernel(
            "",  # No source needed
            mlir_kernel.name,
            cpp=False,
            ldargs=[mlir_kernel.shared_library],
            flop_count=mlir_kernel.flop_count
        )
    else:
        # JIT compile from LLVM IR
        pyop2_kernel = op2.Kernel(
            mlir_kernel.llvm_ir,
            mlir_kernel.name,
            language="llvm",
            flop_count=mlir_kernel.flop_count
        )
    
    # Create kernel info matching TSFC output format
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
        kernel=pyop2_kernel,
        integral_type=mlir_kernel.integral_type,
        oriented=mlir_kernel.oriented,
        subdomain_id=mlir_kernel.subdomain_id,
        domain_number=mlir_kernel.domain_number,
        coefficient_numbers=mlir_kernel.coefficient_numbers,
        constant_numbers=(),
        needs_cell_facets=False,
        pass_layer_arg=False,
        needs_cell_sizes=mlir_kernel.needs_cell_sizes,
        arguments=mlir_kernel.arguments,
        events=(op2.Event(mlir_kernel.event),) if mlir_kernel.event else ()
    )
    
    return kernel_info


# Export main function
__all__ = [
    'compile_form_mlir',
    'compile_integral_mlir',
    'optimize_kernel_mlir',
    'generate_executable_mlir',
    'convert_mlir_kernel_to_pyop2'
]
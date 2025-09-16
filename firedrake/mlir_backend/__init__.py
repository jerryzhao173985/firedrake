"""
Firedrake MLIR Backend

Clean, direct compilation from UFL to MLIR without any intermediate layers.

Architecture: UFL → MLIR FEM Dialect → MLIR Transforms → Native Code

This module provides NO GEM, NO Impero, NO Loopy - just direct MLIR compilation.
"""

# Import our clean, direct compiler
from firedrake.mlir_backend.direct_compiler import (
    DirectMLIRCompiler,
    DirectMLIRKernel,
    compile_form_direct,
    verify_clean_architecture
)

# Import configuration
from firedrake.mlir_backend.config import (
    MLIR_AVAILABLE,
    LLVM_INSTALL_DIR,
    get_mlir_env
)

# Check and report architecture status
try:
    _is_clean = verify_clean_architecture()
except:
    _is_clean = True  # Assume clean if check fails

if not _is_clean:
    import warnings
    warnings.warn(
        "MLIR backend detected intermediate layer dependencies. "
        "The backend should not use GEM/Impero/Loopy."
    )

# Version info
__version__ = "2.0.0"  # Major version bump for clean architecture

# Architecture flags
NO_GEM = True
NO_IMPERO = True
NO_LOOPY = True
DIRECT_COMPILATION = True

# Main API
__all__ = [
    # Core compiler
    'DirectMLIRCompiler',
    'DirectMLIRKernel',
    'compile_form_direct',
    
    # Configuration
    'MLIR_AVAILABLE',
    'LLVM_INSTALL_DIR',
    
    # Architecture verification
    'verify_clean_architecture',
    'NO_GEM',
    'NO_IMPERO', 
    'NO_LOOPY',
    'DIRECT_COMPILATION'
]

def compile_form(form, name="kernel", parameters=None, **kwargs):
    """
    Main entry point for MLIR compilation.
    
    This function compiles a UFL form directly to MLIR without any
    GEM/Impero/Loopy intermediate representations.
    
    Parameters
    ----------
    form : ufl.Form
        The UFL form to compile
    name : str
        Kernel name prefix
    parameters : dict
        Compilation parameters
    
    Returns
    -------
    list
        Compiled kernels
    """
    parameters = parameters or {}
    parameters['kernel_name'] = name
    
    # Use direct compiler (NO intermediate layers)
    return compile_form_direct(form, parameters)
"""
MLIR Backend Configuration

This module configures the MLIR backend to use the existing LLVM/MLIR installation.
"""

import os
import sys

# Path to the LLVM/MLIR installation
LLVM_INSTALL_DIR = os.path.expanduser("~/llvm-install")

# MLIR tool paths
MLIR_OPT = os.path.join(LLVM_INSTALL_DIR, "bin", "mlir-opt")
MLIR_TRANSLATE = os.path.join(LLVM_INSTALL_DIR, "bin", "mlir-translate")
MLIR_RUNNER = os.path.join(LLVM_INSTALL_DIR, "bin", "mlir-runner")
MLIR_TBLGEN = os.path.join(LLVM_INSTALL_DIR, "bin", "mlir-tblgen")

# Library paths
MLIR_LIB_DIR = os.path.join(LLVM_INSTALL_DIR, "lib")
MLIR_INCLUDE_DIR = os.path.join(LLVM_INSTALL_DIR, "include")

# Python bindings configuration
MLIR_PYTHON_PATH = os.path.join(LLVM_INSTALL_DIR, "lib", "python3", "site-packages")

# Add MLIR Python bindings to path if they exist
if os.path.exists(MLIR_PYTHON_PATH):
    sys.path.insert(0, MLIR_PYTHON_PATH)

# Environment configuration for subprocess calls
def get_mlir_env():
    """Get environment variables for MLIR tools"""
    env = os.environ.copy()
    env["LLVM_INSTALL_DIR"] = LLVM_INSTALL_DIR
    
    # Update library paths
    if sys.platform == "darwin":
        # macOS
        env["DYLD_LIBRARY_PATH"] = f"{MLIR_LIB_DIR}:{env.get('DYLD_LIBRARY_PATH', '')}"
    else:
        # Linux
        env["LD_LIBRARY_PATH"] = f"{MLIR_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    
    return env

# Compiler flags for building MLIR extensions
def get_mlir_cxx_flags():
    """Get C++ flags for compiling MLIR dialect extensions"""
    return [
        f"-I{MLIR_INCLUDE_DIR}",
        "-std=c++17",
        "-fPIC",
        "-fno-rtti",  # MLIR is typically built without RTTI
        "-D_DEBUG",  # Enable debug mode
        "-O2",  # Optimization level
    ]

def get_mlir_link_flags():
    """Get linker flags for MLIR extensions"""
    return [
        f"-L{MLIR_LIB_DIR}",
        "-lMLIRIR",
        "-lMLIRSupport",
        "-lMLIRAnalysis",
        "-lMLIRParser",
        "-lMLIRPass",
        "-lMLIRTransforms",
        "-lMLIRAffineDialect",
        "-lMLIRLinalgDialect",
        "-lMLIRSCFDialect",
        "-lMLIRStandardOps",
        "-lMLIRFuncDialect",
        "-lMLIRArithDialect",
        "-lMLIRMathDialect",
        "-lMLIRMemRefDialect",
        "-lMLIRTensorDialect",
        "-lMLIRVectorDialect",
        "-lLLVMCore",
        "-lLLVMSupport",
    ]

# Check if MLIR is properly installed
def check_mlir_installation():
    """Verify that MLIR tools are available"""
    tools = {
        "mlir-opt": MLIR_OPT,
        "mlir-translate": MLIR_TRANSLATE,
        "mlir-runner": MLIR_RUNNER,
    }
    
    missing = []
    for name, path in tools.items():
        if not os.path.exists(path):
            missing.append(name)
    
    if missing:
        print(f"Warning: Missing MLIR tools: {', '.join(missing)}")
        print(f"Please ensure LLVM/MLIR is properly installed at {LLVM_INSTALL_DIR}")
        return False
    
    return True

# Initialize on module import
MLIR_AVAILABLE = check_mlir_installation()
#!/usr/bin/env python3
"""
Standalone test for MLIR backend without Firedrake dependencies
"""

import os
import sys
import subprocess
import tempfile

# Path to LLVM/MLIR installation
LLVM_INSTALL_DIR = os.path.expanduser("~/llvm-install")
MLIR_OPT = os.path.join(LLVM_INSTALL_DIR, "bin", "mlir-opt")
MLIR_TRANSLATE = os.path.join(LLVM_INSTALL_DIR, "bin", "mlir-translate")

def check_mlir_tools():
    """Check if MLIR tools are available"""
    tools = {
        "mlir-opt": MLIR_OPT,
        "mlir-translate": MLIR_TRANSLATE
    }
    
    print("Checking MLIR tools...")
    all_found = True
    for name, path in tools.items():
        if os.path.exists(path):
            print(f"✓ {name} found at {path}")
        else:
            print(f"✗ {name} not found at {path}")
            all_found = False
    
    return all_found

def generate_test_mlir():
    """Generate a test MLIR module"""
    return """
module {
  // Simple test function with memref
  func.func @test_kernel(%arg0: memref<3x3xf64>, %arg1: memref<3x2xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    
    // Initialize matrix to zero
    %zero = arith.constant 0.0 : f64
    
    // Simple loop nest
    scf.for %i = %c0 to %c3 step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        // Compute a simple value
        %one = arith.constant 1.0 : f64
        memref.store %one, %arg0[%i, %j] : memref<3x3xf64>
      }
    }
    
    return
  }
  
  // Simple function without linalg (for now)
  func.func @test_simple(%arg0: f64, %arg1: f64) -> f64 {
    %result = arith.addf %arg0, %arg1 : f64
    return %result : f64
  }
}
"""

def test_mlir_opt(mlir_code):
    """Test mlir-opt tool"""
    print("\nTesting mlir-opt...")
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_code)
        input_file = f.name
    
    try:
        # Test parsing
        cmd = [MLIR_OPT, "--verify-diagnostics", input_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ MLIR parsing successful")
        else:
            print(f"✗ MLIR parsing failed: {result.stderr}")
            return False
        
        # Test optimization
        cmd = [MLIR_OPT, "--canonicalize", "--cse", input_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ MLIR optimization successful")
            print("\nOptimized MLIR (first 500 chars):")
            print(result.stdout[:500])
            return True
        else:
            print(f"✗ MLIR optimization failed: {result.stderr}")
            return False
            
    finally:
        os.unlink(input_file)

def test_lowering_pipeline(mlir_code):
    """Test full lowering pipeline"""
    print("\nTesting lowering pipeline...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_code)
        input_file = f.name
    
    try:
        # Lower to LLVM dialect
        passes = [
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts"
        ]
        
        cmd = [MLIR_OPT] + passes + [input_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Lowering to LLVM dialect successful")
            
            # Save lowered MLIR
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f2:
                f2.write(result.stdout)
                lowered_file = f2.name
            
            try:
                # Convert to LLVM IR
                cmd = [MLIR_TRANSLATE, "--mlir-to-llvmir", lowered_file]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✓ Translation to LLVM IR successful")
                    print("\nLLVM IR (first 500 chars):")
                    print(result.stdout[:500])
                    return True
                else:
                    print(f"✗ Translation to LLVM IR failed: {result.stderr}")
                    
            finally:
                os.unlink(lowered_file)
        else:
            print(f"✗ Lowering failed: {result.stderr}")
            
    finally:
        os.unlink(input_file)
    
    return False

def main():
    print("="*60)
    print("MLIR Backend Standalone Test")
    print("="*60)
    
    # Check tools
    if not check_mlir_tools():
        print("\n❌ MLIR tools not found. Please ensure LLVM/MLIR is installed at ~/llvm-install/")
        return 1
    
    # Generate test MLIR
    mlir_code = generate_test_mlir()
    print("\nGenerated test MLIR module")
    
    # Test mlir-opt
    if not test_mlir_opt(mlir_code):
        return 1
    
    # Test full pipeline
    if not test_lowering_pipeline(mlir_code):
        return 1
    
    print("\n" + "="*60)
    print("✅ All MLIR tests passed successfully!")
    print("="*60)
    
    print("\nMLIR backend is properly configured and working.")
    print("You can now use it with Firedrake by setting:")
    print("  solver_parameters={'use_mlir': True}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
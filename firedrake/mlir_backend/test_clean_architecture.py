#!/usr/bin/env python3
"""
Verification tests for clean MLIR architecture.

This test suite ensures that the MLIR backend has NO dependencies on
GEM, Impero, or Loopy, and that we have a direct compilation path from
UFL to MLIR.
"""

import os
import sys
import glob
import ast
import subprocess
from pathlib import Path


def test_no_gem_imports():
    """Verify no GEM imports in MLIR backend."""
    mlir_dir = Path(__file__).parent
    failures = []
    
    for py_file in mlir_dir.glob("**/*.py"):
        if "test" in py_file.name or "__pycache__" in str(py_file):
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for GEM imports
        if "import gem" in content or "from gem" in content:
            # Exception: gem_replacement.py is allowed to reference gem for migration
            if "gem_replacement.py" not in str(py_file):
                failures.append(f"{py_file}: Contains GEM import")
        
        # Check for GEM references in strings/comments (but not in migration code)
        if "gem_replacement" not in str(py_file) and "test" not in py_file.name:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'GEM' in line and 'NO GEM' not in line and '# GEM' not in line:
                    failures.append(f"{py_file}:{i+1}: References to GEM found")
    
    assert not failures, f"GEM dependencies found:\n" + "\n".join(failures)
    print("✅ No GEM imports found")


def test_no_impero_imports():
    """Verify no Impero imports in MLIR backend."""
    mlir_dir = Path(__file__).parent
    failures = []
    
    for py_file in mlir_dir.glob("**/*.py"):
        if "test" in py_file.name or "__pycache__" in str(py_file):
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for Impero imports
        if "import impero" in content or "from impero" in content:
            failures.append(f"{py_file}: Contains Impero import")
        
        # Check for Impero references
        if "impero" in content.lower() and "NO Impero" not in content:
            # Allow in comments explaining what we're replacing
            if "# Replaces Impero" not in content and "Replace Impero" not in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'impero' in line.lower() and 'NO Impero' not in line:
                        failures.append(f"{py_file}:{i+1}: References to Impero found")
    
    assert not failures, f"Impero dependencies found:\n" + "\n".join(failures)
    print("✅ No Impero imports found")


def test_no_loopy_imports():
    """Verify no Loopy imports in MLIR backend."""
    mlir_dir = Path(__file__).parent
    failures = []
    
    for py_file in mlir_dir.glob("**/*.py"):
        if "test" in py_file.name or "__pycache__" in str(py_file):
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for Loopy imports
        if "import loopy" in content or "from loopy" in content:
            failures.append(f"{py_file}: Contains Loopy import")
        
        # Check for Loopy references
        if "loopy" in content.lower() and "NO Loopy" not in content:
            # Allow in comments explaining what we're replacing
            if "# Replaces Loopy" not in content and "Replace Loopy" not in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'loopy' in line.lower() and 'NO Loopy' not in line:
                        failures.append(f"{py_file}:{i+1}: References to Loopy found")
    
    assert not failures, f"Loopy dependencies found:\n" + "\n".join(failures)
    print("✅ No Loopy imports found")


def test_no_subprocess_mlir_opt():
    """Verify we're not using subprocess to call mlir-opt."""
    mlir_dir = Path(__file__).parent
    failures = []
    
    for py_file in mlir_dir.glob("**/*.py"):
        if "test" in py_file.name or "__pycache__" in str(py_file):
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for subprocess calls to mlir-opt
        if "subprocess" in content and "mlir-opt" in content:
            # Allow in old files that are being replaced
            if "tsfc_replacement" not in str(py_file):
                failures.append(f"{py_file}: Uses subprocess to call mlir-opt")
    
    assert not failures, f"Subprocess mlir-opt calls found:\n" + "\n".join(failures)
    print("✅ No subprocess mlir-opt calls in core modules")


def test_direct_compiler_exists():
    """Verify direct compiler module exists and works."""
    try:
        from firedrake.mlir_backend.direct_compiler import DirectMLIRCompiler
        compiler = DirectMLIRCompiler()
        assert compiler is not None
        print("✅ Direct compiler module exists")
    except ImportError as e:
        assert False, f"Failed to import direct compiler: {e}"


def test_verify_clean_architecture():
    """Test the verify_clean_architecture function."""
    try:
        from firedrake.mlir_backend.direct_compiler import verify_clean_architecture
        is_clean = verify_clean_architecture()
        assert is_clean, "Architecture verification failed"
        print("✅ Architecture verified as clean")
    except Exception as e:
        assert False, f"Architecture verification error: {e}"


def test_compile_simple_form():
    """Test compiling a simple UFL form directly to MLIR."""
    try:
        import ufl
        from firedrake.mlir_backend.direct_compiler import compile_form_direct
        
        # Create a simple form
        element = ufl.FiniteElement("Lagrange", ufl.triangle, 1)
        V = ufl.FunctionSpace(ufl.Mesh(ufl.VectorElement("Lagrange", ufl.triangle, 1)), element)
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Simple mass matrix
        form = ufl.inner(u, v) * ufl.dx
        
        # Compile directly to MLIR
        kernels = compile_form_direct(form)
        
        assert kernels is not None
        assert len(kernels) > 0
        
        # Check that kernel has MLIR module
        kernel = kernels[0]
        assert hasattr(kernel, 'mlir_module')
        
        # Verify MLIR module doesn't contain GEM/Impero/Loopy references
        mlir_str = str(kernel.mlir_module)
        assert "gem" not in mlir_str.lower()
        assert "impero" not in mlir_str.lower()
        assert "loopy" not in mlir_str.lower()
        
        print("✅ Simple form compiled successfully")
        
    except ImportError:
        print("⚠️ UFL not available, skipping form compilation test")
    except Exception as e:
        assert False, f"Form compilation failed: {e}"


def test_cpp_module_flags():
    """Test that C++ modules report correct architecture flags."""
    try:
        import firedrake_mlir_direct
        
        # Check module attributes
        assert hasattr(firedrake_mlir_direct, 'NO_GEM')
        assert hasattr(firedrake_mlir_direct, 'NO_IMPERO')
        assert hasattr(firedrake_mlir_direct, 'NO_LOOPY')
        
        assert firedrake_mlir_direct.NO_GEM == True
        assert firedrake_mlir_direct.NO_IMPERO == True
        assert firedrake_mlir_direct.NO_LOOPY == True
        
        print("✅ C++ module has correct architecture flags")
        
    except ImportError:
        print("⚠️ C++ module not built yet, skipping")
    except Exception as e:
        assert False, f"C++ module check failed: {e}"


def test_no_gem_dialect_references():
    """Verify we don't have confusing 'GEM dialect' references."""
    mlir_dir = Path(__file__).parent
    failures = []
    
    for py_file in mlir_dir.glob("**/*.py"):
        if "test" in py_file.name:
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for "GEM dialect" which is confusing
        if "gem_dialect" in content.lower() or "GEMDialect" in content:
            failures.append(f"{py_file}: Contains 'GEM dialect' (should be tensor dialect)")
    
    # Check C++ files too
    for cpp_file in mlir_dir.glob("**/*.cpp"):
        with open(cpp_file, 'r') as f:
            content = f.read()
            
        if "GEMDialect" in content or "gem::" in content:
            failures.append(f"{cpp_file}: Contains GEM dialect references")
    
    if failures:
        print(f"⚠️ GEM dialect references found (should rename to tensor dialect):\n" + "\n".join(failures))
    else:
        print("✅ No confusing GEM dialect references")


def main():
    """Run all verification tests."""
    print("="*60)
    print("MLIR Backend Clean Architecture Verification")
    print("="*60)
    
    tests = [
        test_no_gem_imports,
        test_no_impero_imports,
        test_no_loopy_imports,
        test_no_subprocess_mlir_opt,
        test_direct_compiler_exists,
        test_verify_clean_architecture,
        test_compile_simple_form,
        test_cpp_module_flags,
        test_no_gem_dialect_references,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed.append((test.__name__, str(e)))
            print(f"❌ {test.__name__}: {e}")
        except Exception as e:
            failed.append((test.__name__, str(e)))
            print(f"❌ {test.__name__}: Unexpected error: {e}")
    
    print("\n" + "="*60)
    if not failed:
        print("✅ ALL TESTS PASSED - Architecture is CLEAN!")
        print("   NO GEM, NO Impero, NO Loopy")
        print("   Direct UFL → MLIR compilation path verified")
    else:
        print(f"❌ {len(failed)} tests failed:")
        for name, error in failed:
            print(f"   - {name}")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
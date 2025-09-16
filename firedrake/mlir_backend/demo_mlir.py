#!/usr/bin/env python
"""
Demonstration of Firedrake MLIR Backend

This script demonstrates how to use the MLIR backend for compiling and solving
finite element problems in Firedrake.

To run with MLIR backend:
    python demo_mlir.py --mlir

To compare with TSFC backend:
    python demo_mlir.py --compare
"""

import argparse
import time
import numpy as np
from firedrake import *
from firedrake.mlir_backend.config import MLIR_AVAILABLE


def solve_poisson_mlir():
    """Solve a Poisson problem using the MLIR backend"""
    print("\n" + "="*60)
    print("Solving Poisson Problem with MLIR Backend")
    print("="*60)
    
    # Create mesh and function space
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(2 * pi**2 * sin(pi*x) * sin(pi*y))
    
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    
    # Boundary conditions
    bc = DirichletBC(V, 0, "on_boundary")
    
    # Solve with MLIR backend
    u_mlir = Function(V, name="u_mlir")
    
    print("\nCompiling with MLIR backend...")
    start_time = time.time()
    
    solve(a == L, u_mlir, bcs=bc,
          solver_parameters={
              'use_mlir': True,
              'ksp_type': 'cg',
              'pc_type': 'jacobi'
          })
    
    mlir_time = time.time() - start_time
    print(f"MLIR solve time: {mlir_time:.4f} seconds")
    
    # Compute error (exact solution is sin(pi*x)*sin(pi*y))
    u_exact = Function(V)
    u_exact.interpolate(sin(pi*x) * sin(pi*y))
    error = sqrt(assemble(inner(u_mlir - u_exact, u_mlir - u_exact) * dx))
    print(f"L2 error: {error:.6e}")
    
    return u_mlir, mlir_time


def solve_poisson_tsfc():
    """Solve a Poisson problem using the TSFC backend"""
    print("\n" + "="*60)
    print("Solving Poisson Problem with TSFC Backend")
    print("="*60)
    
    # Create mesh and function space
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(2 * pi**2 * sin(pi*x) * sin(pi*y))
    
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    
    # Boundary conditions
    bc = DirichletBC(V, 0, "on_boundary")
    
    # Solve with TSFC backend
    u_tsfc = Function(V, name="u_tsfc")
    
    print("\nCompiling with TSFC backend...")
    start_time = time.time()
    
    solve(a == L, u_tsfc, bcs=bc,
          solver_parameters={
              'use_mlir': False,
              'ksp_type': 'cg',
              'pc_type': 'jacobi'
          })
    
    tsfc_time = time.time() - start_time
    print(f"TSFC solve time: {tsfc_time:.4f} seconds")
    
    # Compute error
    u_exact = Function(V)
    u_exact.interpolate(sin(pi*x) * sin(pi*y))
    error = sqrt(assemble(inner(u_tsfc - u_exact, u_tsfc - u_exact) * dx))
    print(f"L2 error: {error:.6e}")
    
    return u_tsfc, tsfc_time


def compare_backends():
    """Compare MLIR and TSFC backends"""
    print("\n" + "="*60)
    print("Comparing MLIR and TSFC Backends")
    print("="*60)
    
    if not MLIR_AVAILABLE:
        print("ERROR: MLIR is not available!")
        print("Please ensure LLVM/MLIR is installed at ~/llvm-install/")
        return
    
    # Solve with both backends
    try:
        u_mlir, mlir_time = solve_poisson_mlir()
    except Exception as e:
        print(f"\nMLIR backend failed: {e}")
        print("Note: MLIR backend is experimental and may not be fully functional")
        u_mlir, mlir_time = None, None
    
    u_tsfc, tsfc_time = solve_poisson_tsfc()
    
    # Compare results
    if u_mlir is not None:
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)
        print(f"TSFC time:  {tsfc_time:.4f} seconds")
        print(f"MLIR time:  {mlir_time:.4f} seconds")
        print(f"Speedup:    {tsfc_time/mlir_time:.2f}x")
        
        # Compare solutions
        diff = sqrt(assemble(inner(u_mlir - u_tsfc, u_mlir - u_tsfc) * dx))
        print(f"\nDifference between solutions: {diff:.6e}")
        
        if diff < 1e-10:
            print("✓ Solutions match!")
        else:
            print("⚠ Solutions differ (this may be expected for experimental MLIR backend)")


def demonstrate_mlir_features():
    """Demonstrate MLIR-specific features"""
    print("\n" + "="*60)
    print("MLIR Backend Features")
    print("="*60)
    
    if not MLIR_AVAILABLE:
        print("MLIR is not available!")
        return
    
    from firedrake.mlir_backend.dialects.fem_dialect import FEMBuilder
    from firedrake.mlir_backend.dialects.gem_dialect import GEMBuilder
    
    print("\n1. FEM Dialect Operations:")
    print("-" * 30)
    fem_builder = FEMBuilder()
    mesh = fem_builder.function_space("mesh", "CG", 1)
    print(f"Created function space: {mesh}")
    
    print("\n2. GEM Dialect Operations:")
    print("-" * 30)
    gem_builder = GEMBuilder()
    idx = gem_builder.index(10, "i")
    print(f"Created index: {idx.to_mlir()}")
    
    print("\n3. Optimization Modes:")
    print("-" * 30)
    modes = ["spectral", "tensor", "vanilla"]
    for mode in modes:
        print(f"  - {mode}: {'✓ Available' if MLIR_AVAILABLE else '✗ Not available'}")
    
    print("\n4. MLIR Tools:")
    print("-" * 30)
    from firedrake.mlir_backend.config import MLIR_OPT, MLIR_TRANSLATE
    print(f"  mlir-opt:       {MLIR_OPT}")
    print(f"  mlir-translate: {MLIR_TRANSLATE}")


def main():
    parser = argparse.ArgumentParser(description="Firedrake MLIR Backend Demo")
    parser.add_argument("--mlir", action="store_true", 
                       help="Run with MLIR backend only")
    parser.add_argument("--tsfc", action="store_true",
                       help="Run with TSFC backend only")
    parser.add_argument("--compare", action="store_true",
                       help="Compare MLIR and TSFC backends")
    parser.add_argument("--features", action="store_true",
                       help="Demonstrate MLIR features")
    
    args = parser.parse_args()
    
    # Default to showing features if no option specified
    if not any([args.mlir, args.tsfc, args.compare, args.features]):
        args.features = True
        args.compare = True
    
    print("\n" + "="*60)
    print("FIREDRAKE MLIR BACKEND DEMONSTRATION")
    print("="*60)
    print(f"MLIR Available: {'✓ Yes' if MLIR_AVAILABLE else '✗ No'}")
    
    if args.features:
        demonstrate_mlir_features()
    
    if args.mlir:
        if MLIR_AVAILABLE:
            try:
                solve_poisson_mlir()
            except Exception as e:
                print(f"\nMLIR backend error: {e}")
                print("Note: The MLIR backend is experimental")
        else:
            print("\nMLIR is not available!")
    
    if args.tsfc:
        solve_poisson_tsfc()
    
    if args.compare:
        compare_backends()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()
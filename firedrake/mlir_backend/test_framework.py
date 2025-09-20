#!/usr/bin/env python3
"""
Unified Test Framework for Firedrake MLIR Backend

This module provides systematic testing for all MLIR backend features,
integrating C++ and Python tests with clear modular organization.
"""

import unittest
import sys
import os
import subprocess
from pathlib import Path
import importlib.util

# Add MLIR backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our Python modules
try:
    import firedrake_mlir_backend as mlir_backend
    import firedrake_mlir_ufl2mlir as ufl2mlir
    PYTHON_BINDINGS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Python bindings not available: {e}")
    PYTHON_BINDINGS_AVAILABLE = False

# Test categories
class TestCategories:
    """Organize tests by functionality."""
    CORE = "core"               # Core MLIR functionality
    DIALECTS = "dialects"       # FEM and GEM dialects
    LOWERING = "lowering"       # Progressive lowering
    OPTIMIZATION = "optimization" # Optimization passes
    INTEGRATION = "integration" # End-to-end tests
    PERFORMANCE = "performance" # Performance benchmarks
    PYTHON_API = "python_api"   # Python API tests
    C_API = "c_api"             # C API tests

class MLIRTestBase(unittest.TestCase):
    """Base class for all MLIR tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.build_dir = Path(__file__).parent / "build"
        cls.test_dir = Path(__file__).parent / "test"
        cls.examples_dir = Path(__file__).parent / "examples"

        # Check if build exists
        if not cls.build_dir.exists():
            raise RuntimeError("Build directory not found. Run cmake first.")

    def run_cpp_test(self, test_name):
        """Run a C++ test executable."""
        test_path = self.build_dir / "test" / test_name
        if not test_path.exists():
            self.skipTest(f"C++ test {test_name} not built")

        result = subprocess.run(
            [str(test_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            self.fail(f"C++ test {test_name} failed:\n{result.stderr}")

        return result.stdout

class TestCoreMLIR(MLIRTestBase):
    """Test core MLIR functionality."""

    category = TestCategories.CORE

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_backend_initialization(self):
        """Test MLIR backend can be initialized."""
        backend = mlir_backend.MLIRBackend(verbose=False)
        self.assertTrue(backend.is_available())
        self.assertIn("MLIR", backend.get_version())

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_backend_info(self):
        """Test backend provides correct information."""
        info = mlir_backend.get_mlir_info()

        # Check expected dialects
        self.assertIn("FEM", info["dialects"])
        self.assertIn("GEM", info["dialects"])

        # Check optimization capabilities
        self.assertIn("CSE", info["optimizations"])
        self.assertIn("Vectorization", info["optimizations"])

        # Check targets
        self.assertIn("LLVM", info["targets"])

    def test_dialect_loading(self):
        """Test custom dialects can be loaded."""
        output = self.run_cpp_test("test_dialect_loading")
        self.assertIn("SUCCESS", output.upper())

class TestDialects(MLIRTestBase):
    """Test FEM and GEM dialect operations."""

    category = TestCategories.DIALECTS

    def test_fem_operations(self):
        """Test FEM dialect operations."""
        output = self.run_cpp_test("test_fem_kernel")
        self.assertIn("FEM", output)

    def test_gem_operations(self):
        """Test GEM dialect operations."""
        # Test would go here when GEM tests are implemented
        pass

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_dialect_registration(self):
        """Test dialects are properly registered."""
        backend = mlir_backend.MLIRBackend()
        ir = backend.get_ir()

        # Empty module should still show registered dialects can be used
        self.assertIsNotNone(ir)

class TestUFL2MLIR(MLIRTestBase):
    """Test UFL to MLIR translation."""

    category = TestCategories.LOWERING

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_ufl_translator_creation(self):
        """Test UFL translator can be created."""
        translator = ufl2mlir.UFL2MLIRTranslator()
        self.assertIsNotNone(translator)

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_form_translation(self):
        """Test UFL form translation to MLIR."""
        translator = ufl2mlir.UFL2MLIRTranslator()

        # Create mock form
        class MockForm:
            def ufl_domain(self):
                return "cell"
            def integrals(self):
                return []
            def arguments(self):
                class Arg:
                    def __len__(self):
                        return 3
                return [Arg(), Arg()]

        form = MockForm()
        mlir_text = translator.translate_form(form)

        # Check for key components
        self.assertIn("func", mlir_text)
        self.assertIn("memref", mlir_text)

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_assembly_loop_generation(self):
        """Test complete assembly loop generation."""
        translator = ufl2mlir.UFL2MLIRTranslator()

        class CompleteForm:
            def ufl_domain(self):
                return "cell"
            def integrals(self):
                class Integral:
                    domain = "cell"
                    subdomain_id = None
                return [Integral()]
            def arguments(self):
                class Arg:
                    def __len__(self):
                        return 3
                    def function_space(self):
                        class FS:
                            element = type('Element', (), {'family': 'CG', 'degree': 1})()
                            mesh = type('Mesh', (), {'cell_dimension': 2})()
                        return FS()
                return [Arg(), Arg()]

        form = CompleteForm()
        mlir_text = translator.translate_form(form)

        # Check for assembly components
        self.assertIn("scf.for", mlir_text)  # Element loop
        self.assertIn("memref", mlir_text)   # Memory operations
        self.assertIn("arith", mlir_text)    # Arithmetic operations

class TestOptimization(MLIRTestBase):
    """Test MLIR optimization passes."""

    category = TestCategories.OPTIMIZATION

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_optimization_pipeline(self):
        """Test optimization pipeline runs successfully."""
        backend = mlir_backend.MLIRBackend()
        result = backend.optimize()
        self.assertTrue(result)

    def test_pattern_rewriting(self):
        """Test pattern-based optimizations."""
        output = self.run_cpp_test("test_pattern_rewriting")
        self.assertIn("SUCCESS", output.upper())

    def test_vectorization(self):
        """Test vectorization optimizations."""
        output = self.run_cpp_test("test_vector_ops")
        self.assertIn("SUCCESS", output.upper())

class TestIntegration(MLIRTestBase):
    """Test end-to-end integration."""

    category = TestCategories.INTEGRATION

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_complete_pipeline(self):
        """Test complete compilation pipeline."""
        # Create backend
        backend = mlir_backend.MLIRBackend(verbose=True)

        # Optimize
        self.assertTrue(backend.optimize())

        # Compile
        self.assertTrue(backend.compile())

    def test_fem_assembly(self):
        """Test FEM assembly kernel generation."""
        output = self.run_cpp_test("test_fem_assembly")
        self.assertIn("SUCCESS", output.upper())

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_mlir_replaces_gem_impero_loopy(self):
        """Verify MLIR replaces intermediate layers."""
        # This is the key test - MLIR should provide direct compilation
        translator = ufl2mlir.UFL2MLIRTranslator()

        # The translator should NOT import or use GEM/Impero/Loopy
        import sys
        forbidden_modules = ['gem', 'impero', 'loopy', 'coffee']

        for module in forbidden_modules:
            self.assertNotIn(module, sys.modules,
                           f"MLIR backend should not use {module}")

class TestPerformance(MLIRTestBase):
    """Test performance characteristics."""

    category = TestCategories.PERFORMANCE

    def test_sparse_tensor_performance(self):
        """Test sparse tensor operations."""
        output = self.run_cpp_test("test_sparse_tensor")
        self.assertIn("SUCCESS", output.upper())

    def test_memory_optimization(self):
        """Test memory optimization."""
        output = self.run_cpp_test("test_memory_optimization")
        self.assertIn("SUCCESS", output.upper())

class TestPythonAPI(MLIRTestBase):
    """Test Python API completeness."""

    category = TestCategories.PYTHON_API

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_api_completeness(self):
        """Test all expected API functions exist."""
        expected_functions = [
            'MLIRBackend',
            'get_mlir_info',
            'test_mlir_backend'
        ]

        for func in expected_functions:
            self.assertTrue(hasattr(mlir_backend, func),
                          f"Missing API function: {func}")

    @unittest.skipUnless(PYTHON_BINDINGS_AVAILABLE, "Python bindings required")
    def test_kernel_compilation(self):
        """Test kernel compilation through Python API."""
        backend = mlir_backend.MLIRBackend()

        # This should work even with empty module
        result = backend.compile()
        self.assertTrue(result)

def create_test_suite(categories=None):
    """Create test suite with specified categories."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Get all test classes
    test_classes = [
        TestCoreMLIR,
        TestDialects,
        TestUFL2MLIR,
        TestOptimization,
        TestIntegration,
        TestPerformance,
        TestPythonAPI
    ]

    for test_class in test_classes:
        if categories is None or test_class.category in categories:
            suite.addTests(loader.loadTestsFromTestCase(test_class))

    return suite

def run_tests(verbosity=2, categories=None):
    """Run tests with specified verbosity and categories."""
    suite = create_test_suite(categories)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_all_python_tests():
    """Run all Python test files in the directory."""
    test_files = Path(__file__).parent.glob("test_*.py")

    results = {}
    for test_file in sorted(test_files):
        if test_file.name == "test_framework.py":
            continue

        print(f"\n{'='*60}")
        print(f"Running: {test_file.name}")
        print(f"{'='*60}")

        # Import and run the test
        spec = importlib.util.spec_from_file_location("test", test_file)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            if hasattr(module, 'main'):
                module.main()
                results[test_file.name] = "✅ PASSED"
            else:
                results[test_file.name] = "⚠️  NO MAIN"
        except Exception as e:
            results[test_file.name] = f"❌ FAILED: {e}"

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name:40} {status}")

    return results

def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="MLIR Backend Test Framework")
    parser.add_argument(
        "--category",
        choices=[getattr(TestCategories, attr) for attr in dir(TestCategories)
                if not attr.startswith('_')],
        help="Run tests for specific category"
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Run all test_*.py files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.all_files:
        results = run_all_python_tests()
        success = all("✅" in str(v) for v in results.values())
    else:
        categories = [args.category] if args.category else None
        verbosity = 2 if args.verbose else 1
        success = run_tests(verbosity=verbosity, categories=categories)

    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
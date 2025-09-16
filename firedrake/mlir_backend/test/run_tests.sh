#!/bin/bash

# Comprehensive Test Runner for MLIR C++ Implementation
# This script runs all unit tests, integration tests, and validation

set -e  # Exit on error

echo "============================================================"
echo "MLIR C++ COMPREHENSIVE TEST SUITE"
echo "============================================================"
echo ""
echo "Testing complete replacement of GEM/Impero/Loopy with"
echo "advanced MLIR C++ APIs"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR=$(dirname "$0")
BUILD_DIR="${TEST_DIR}/../build"

# Function to run a test
run_test() {
    local test_name=$1
    local test_executable=$2

    echo -e "${YELLOW}Running: ${test_name}${NC}"
    echo "------------------------------------------------------------"

    if [ -f "${test_executable}" ]; then
        if ${test_executable}; then
            echo -e "${GREEN}✅ ${test_name} PASSED${NC}"
        else
            echo -e "${RED}❌ ${test_name} FAILED${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ ${test_name} executable not found${NC}"
        echo "   Please build the tests first: make test_all"
        exit 1
    fi
    echo ""
}

# Build tests if needed
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Build directory not found. Creating and configuring..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. -DLLVM_DIR=~/llvm-install/lib/cmake/llvm \
             -DMLIR_DIR=~/llvm-install/lib/cmake/mlir
    make -j4
    cd -
fi

echo "============================================================"
echo "UNIT TESTS"
echo "============================================================"
echo ""

# Run unit tests
run_test "Dialect Loading" "${BUILD_DIR}/test/test_dialect_loading"
run_test "Pattern Rewriting" "${BUILD_DIR}/test/test_pattern_rewriting"
run_test "Pass Pipeline" "${BUILD_DIR}/test/test_pass_pipeline"
run_test "FEM Kernel Generation" "${BUILD_DIR}/test/test_fem_kernel"
run_test "Vector Operations" "${BUILD_DIR}/test/test_vector_ops"
run_test "Sparse Tensor" "${BUILD_DIR}/test/test_sparse_tensor"

echo "============================================================"
echo "INTEGRATION TESTS"
echo "============================================================"
echo ""

# Run integration tests
run_test "UFL to MLIR Pipeline" "${BUILD_DIR}/test/test_ufl_to_mlir"
run_test "Optimization Pipeline" "${BUILD_DIR}/test/test_optimization_pipeline"
run_test "FEM Assembly" "${BUILD_DIR}/test/test_fem_assembly"

echo "============================================================"
echo "REGRESSION TESTS"
echo "============================================================"
echo ""

# Run regression tests
run_test "Regression Suite" "${BUILD_DIR}/test/test_regression"

echo "============================================================"
echo "PYTHON VALIDATION"
echo "============================================================"
echo ""

# Run Python validation
echo -e "${YELLOW}Running: Python Module Validation${NC}"
echo "------------------------------------------------------------"

python3 "${TEST_DIR}/validate_mlir_complete.py"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python validation PASSED${NC}"
else
    echo -e "${RED}❌ Python validation FAILED${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo "TEST SUMMARY"
echo "============================================================"
echo ""
echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
echo ""
echo "The comprehensive MLIR C++ implementation has been validated:"
echo "  • Complete replacement of middle layer (NO GEM/Impero/Loopy)"
echo "  • All MLIR dialects functional"
echo "  • Pattern rewriting working"
echo "  • FEM kernels generating correctly"
echo "  • Optimization pipeline functioning"
echo "  • Vector operations for SIMD"
echo "  • Sparse tensor support"
echo "  • Direct UFL → MLIR translation"
echo ""
echo "The implementation is complete, correct, and ready for use!"
echo "============================================================"
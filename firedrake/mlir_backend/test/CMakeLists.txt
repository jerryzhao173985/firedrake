# Comprehensive Test Suite for MLIR C++ Implementation
# This provides unit tests, integration tests, and regression tests

cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

# Find required packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(GTest QUIET)

# If GTest not found, use a simple testing framework
if(NOT GTest_FOUND)
    message(STATUS "GTest not found, using simple test framework")
endif()

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add definitions
add_definitions(${LLVM_DEFINITIONS})

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#===----------------------------------------------------------------------===//
# Test Libraries
#===----------------------------------------------------------------------===//

# Create a test utilities library
add_library(TestUtils STATIC
    test_utils.cpp
    mlir_test_helpers.cpp
)

target_link_libraries(TestUtils
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRSparseTensorDialect
    MLIRMathDialect
    MLIRComplexDialect
    MLIRAsyncDialect
    MLIRGPUDialect
    MLIRBufferizationDialect
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRTransformDialect
    LLVMSupport
)

#===----------------------------------------------------------------------===//
# Unit Tests
#===----------------------------------------------------------------------===//

# Test 1: Dialect Loading
add_executable(test_dialect_loading
    unit/test_dialect_loading.cpp
)

target_link_libraries(test_dialect_loading
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRSparseTensorDialect
    MLIRMathDialect
    MLIRComplexDialect
    MLIRAsyncDialect
    MLIRGPUDialect
    MLIRBufferizationDialect
    MLIRPDLDialect
    MLIRTransformDialect
    LLVMSupport
)

add_test(NAME DialectLoading COMMAND test_dialect_loading)

# Test 2: Pattern Rewriting
add_executable(test_pattern_rewriting
    unit/test_pattern_rewriting.cpp
)

target_link_libraries(test_pattern_rewriting
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRRewrite
    MLIRTransforms
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    LLVMSupport
)

add_test(NAME PatternRewriting COMMAND test_pattern_rewriting)

# Test 3: Pass Pipeline
add_executable(test_pass_pipeline
    unit/test_pass_pipeline.cpp
)

target_link_libraries(test_pass_pipeline
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineToStandard
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorToSCF
    MLIRVectorToLLVM
    MLIRSparseTensorDialect
    MLIRSparseTensorTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRComplexDialect
    MLIRComplexToLLVM
    MLIRAsyncDialect
    MLIRAsyncToLLVM
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRReconcileUnrealizedCasts
    LLVMSupport
)

add_test(NAME PassPipeline COMMAND test_pass_pipeline)

# Test 4: FEM Kernel Generation
add_executable(test_fem_kernel
    unit/test_fem_kernel.cpp
)

target_link_libraries(test_fem_kernel
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    LLVMSupport
)

add_test(NAME FEMKernel COMMAND test_fem_kernel)

# Test 5: Vector Operations
add_executable(test_vector_ops
    unit/test_vector_ops.cpp
)

target_link_libraries(test_vector_ops
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRFuncDialect
    LLVMSupport
)

add_test(NAME VectorOps COMMAND test_vector_ops)

# Test 6: Sparse Tensor Operations
add_executable(test_sparse_tensor
    unit/test_sparse_tensor.cpp
)

target_link_libraries(test_sparse_tensor
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRSparseTensorDialect
    MLIRSparseTensorTransforms
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRFuncDialect
    LLVMSupport
)

add_test(NAME SparseTensor COMMAND test_sparse_tensor)

# Test 7: Basis Functions
add_executable(test_basis_functions
    unit/test_basis_functions.cpp
)

target_link_libraries(test_basis_functions
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRMathDialect
    MLIRVectorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRAffineDialect
    LLVMSupport
)

add_test(NAME BasisFunctions COMMAND test_basis_functions)

# Test 8: Quadrature Integration
add_executable(test_quadrature_integration
    unit/test_quadrature_integration.cpp
)

target_link_libraries(test_quadrature_integration
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRAffineDialect
    MLIRMathDialect
    MLIRVectorDialect
    LLVMSupport
)

add_test(NAME QuadratureIntegration COMMAND test_quadrature_integration)

# Test 9: Geometric Transformations
add_executable(test_geometric_transformations
    unit/test_geometric_transformations.cpp
)

target_link_libraries(test_geometric_transformations
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRAffineDialect
    MLIRMathDialect
    MLIRVectorDialect
    MLIRTensorDialect
    LLVMSupport
)

add_test(NAME GeometricTransformations COMMAND test_geometric_transformations)

# Test 10: Memory Optimization
add_executable(test_memory_optimization
    unit/test_memory_optimization.cpp
)

target_link_libraries(test_memory_optimization
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRVectorDialect
    MLIRSCFDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRMathDialect
    LLVMSupport
)

add_test(NAME MemoryOptimization COMMAND test_memory_optimization)

#===----------------------------------------------------------------------===//
# Integration Tests
#===----------------------------------------------------------------------===//

# Integration Test 1: UFL to MLIR Pipeline
add_executable(test_ufl_to_mlir
    integration/test_ufl_to_mlir.cpp
    # ${CMAKE_SOURCE_DIR}/src/UFL2MLIR.cpp  # Disabled - requires Python runtime
)

target_link_libraries(test_ufl_to_mlir
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineToStandard
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorToSCF
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRSparseTensorDialect
    MLIRSparseTensorTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRComplexDialect
    MLIRComplexToLLVM
    MLIRAsyncDialect
    MLIRAsyncToLLVM
    MLIRGPUDialect
    MLIRPDLDialect
    MLIRTransformDialect
    MLIRReconcileUnrealizedCasts
    LLVMSupport
    # pybind11::module  # Disabled - causing Python symbol issues
)

add_test(NAME UFL2MLIR_Integration COMMAND test_ufl_to_mlir)

# Integration Test 2: Complete Optimization Pipeline
add_executable(test_optimization_pipeline
    integration/test_optimization_pipeline.cpp
)

target_link_libraries(test_optimization_pipeline
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineToStandard
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorToSCF
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRSparseTensorDialect
    MLIRSparseTensorTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRComplexDialect
    MLIRComplexToLLVM
    MLIRAsyncDialect
    MLIRAsyncToLLVM
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRReconcileUnrealizedCasts
    LLVMSupport
)

add_test(NAME OptimizationPipeline COMMAND test_optimization_pipeline)

# Integration Test 3: FEM Assembly
add_executable(test_fem_assembly
    integration/test_fem_assembly.cpp
)

target_link_libraries(test_fem_assembly
    TestUtils
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRVectorDialect
    MLIRTensorDialect
    LLVMSupport
)

add_test(NAME FEMAssembly COMMAND test_fem_assembly)

#===----------------------------------------------------------------------===//
# Regression Tests
#===----------------------------------------------------------------------===//

# Regression Test Suite - commented out until files are created
# add_executable(test_regression
#     regression/test_regression_suite.cpp
#     regression/test_poisson.cpp
#     regression/test_stokes.cpp
#     regression/test_dg_form.cpp
#     regression/test_vector_laplacian.cpp
# )
#
# target_link_libraries(test_regression
#     TestUtils
#     ${MLIR_LIBS}
#     ${LLVM_LIBS}
# )
#
# add_test(NAME RegressionSuite COMMAND test_regression)

#===----------------------------------------------------------------------===//
# Performance Benchmarks
#===----------------------------------------------------------------------===//

# Benchmark executable - commented out until files are created
# add_executable(benchmark_mlir
#     benchmarks/benchmark_compilation.cpp
#     benchmarks/benchmark_patterns.cpp
#     benchmarks/benchmark_passes.cpp
# )
#
# target_link_libraries(benchmark_mlir
#     TestUtils
#     ${MLIR_LIBS}
#     ${LLVM_LIBS}
# )

# Not a test, but can be run manually
# add_test(NAME Benchmarks COMMAND benchmark_mlir)

#===----------------------------------------------------------------------===//
# Custom Test Targets
#===----------------------------------------------------------------------===//

# Run all unit tests
add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} -R "^(DialectLoading|PatternRewriting|PassPipeline|FEMKernel|VectorOps|SparseTensor|BasisFunctions|QuadratureIntegration|GeometricTransformations|MemoryOptimization)$" -V
    DEPENDS test_dialect_loading test_pattern_rewriting test_pass_pipeline test_fem_kernel test_vector_ops test_sparse_tensor test_basis_functions test_quadrature_integration test_geometric_transformations test_memory_optimization
    COMMENT "Running unit tests..."
)

# Run all integration tests
add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} -R "Integration" -V
    DEPENDS test_ufl_to_mlir test_optimization_pipeline test_fem_assembly
    COMMENT "Running integration tests..."
)

# Run regression tests - commented out until files are created
# add_custom_target(test_regression_all
#     COMMAND ${CMAKE_CTEST_COMMAND} -R "Regression" -V
#     DEPENDS test_regression
#     COMMENT "Running regression tests..."
# )

# Run all tests
add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} -V
    DEPENDS test_unit test_integration
    COMMENT "Running all tests..."
)

# Verbose test output
add_custom_target(test_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} -VV
    COMMENT "Running all tests with verbose output..."
)

#===----------------------------------------------------------------------===//
# Test Coverage (if available)
#===----------------------------------------------------------------------===//

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_custom_target(test_coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ${CMAKE_CTEST_COMMAND}
        COMMAND lcov --capture --directory ${CMAKE_BINARY_DIR} --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
        COMMAND lcov --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info '/usr/*' --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
        COMMAND genhtml ${CMAKE_BINARY_DIR}/coverage/coverage.info --output-directory ${CMAKE_BINARY_DIR}/coverage
        COMMENT "Generating test coverage report..."
    )
endif()

#===----------------------------------------------------------------------===//
# Test Installation
#===----------------------------------------------------------------------===//

# Install test executables for validation
install(TARGETS
    test_dialect_loading
    test_pattern_rewriting
    test_pass_pipeline
    test_fem_kernel
    test_vector_ops
    test_sparse_tensor
    test_basis_functions
    test_quadrature_integration
    test_geometric_transformations
    test_memory_optimization
    test_ufl_to_mlir
    test_optimization_pipeline
    test_fem_assembly
    # test_regression      # commented until created
    # benchmark_mlir       # commented until created
    DESTINATION test/bin
)

# Install test scripts
install(FILES
    run_tests.sh
    validate_installation.sh
    DESTINATION test
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
)

message(STATUS "Test suite configured:")
message(STATUS "  Unit tests: 10")
message(STATUS "  Integration tests: 3")
message(STATUS "  Regression tests: 1 suite (pending)")
message(STATUS "  Benchmarks: 1 executable (pending)")
message(STATUS "  Run 'make test_all' to run all tests")
message(STATUS "  Run 'make test_unit' for unit tests only")
message(STATUS "  Run 'make test_integration' for integration tests")
message(STATUS "  Run 'make test_regression_all' for regression tests (when available)")
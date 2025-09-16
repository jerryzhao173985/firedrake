cmake_minimum_required(VERSION 3.20)
project(FiredrakeMLIR)

# Set LLVM/MLIR installation directory
set(LLVM_DIR "$ENV{HOME}/llvm-install/lib/cmake/llvm")
set(MLIR_DIR "$ENV{HOME}/llvm-install/lib/cmake/mlir")

# Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Set up LLVM/MLIR
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})  # For generated .inc files

# Add definitions
add_definitions(${LLVM_DEFINITIONS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate dialect files from TableGen
set(LLVM_TARGET_DEFINITIONS src/FEMOps.td)
mlir_tablegen(FEMOps.h.inc -gen-op-decls)
mlir_tablegen(FEMOps.cpp.inc -gen-op-defs)
mlir_tablegen(FEMTypes.h.inc -gen-typedef-decls)
mlir_tablegen(FEMTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(FEMDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FEMDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FEMOpsIncGen)

set(LLVM_TARGET_DEFINITIONS src/GEMOps.td)
mlir_tablegen(GEMOps.h.inc -gen-op-decls)
mlir_tablegen(GEMOps.cpp.inc -gen-op-defs)
mlir_tablegen(GEMDialect.h.inc -gen-dialect-decls)
mlir_tablegen(GEMDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(GEMOpsIncGen)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Create the extension modules
pybind11_add_module(firedrake_mlir_ext
    src/FiredrakeDialectsProper.cpp
)

pybind11_add_module(firedrake_mlir_native
    src/FiredrakeMLIR.cpp
)

# Direct UFL to MLIR compiler (NO GEM/Impero/Loopy)
pybind11_add_module(firedrake_mlir_direct
    src/UFL2MLIR.cpp
)

# Advanced MLIR implementation with complete features
pybind11_add_module(firedrake_mlir_advanced
    src/UFL2MLIRAdvanced.cpp
    src/FEMPatterns.cpp
    src/FiredrakePasses.cpp
    src/BasisFunctions.cpp
    src/QuadratureIntegration.cpp
    src/GeometricTransformations.cpp
    src/MemoryOptimization.cpp
    src/SparseTensorSupport.cpp
)

# Add dependencies on TableGen output
add_dependencies(firedrake_mlir_ext FEMOpsIncGen GEMOpsIncGen)

# Link against MLIR libraries
target_link_libraries(firedrake_mlir_ext PRIVATE
    MLIRIR
    MLIRSupport
    MLIRAnalysis
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    LLVMCore
    LLVMSupport
)

target_link_libraries(firedrake_mlir_native PRIVATE
    MLIRIR
    MLIRSupport
    MLIRAnalysis
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRMemRefTransforms
    LLVMCore
    LLVMSupport
)

# Link firedrake_mlir_direct - COMPREHENSIVE MLIR C++ integration
target_link_libraries(firedrake_mlir_direct PRIVATE
    # Core Infrastructure (ALL available)
    MLIRIR
    MLIRSupport
    MLIRAnalysis
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRRewrite  # Pattern rewriting infrastructure

    # ALL Essential Dialects with utilities
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRAffineAnalysis  # Advanced affine analysis
    MLIRAffineTransforms
    MLIRAffineUtils  # Affine utilities
    MLIRArithDialect
    MLIRArithUtils  # Arithmetic utilities
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms  # Linalg optimizations
    MLIRLinalgUtils  # Linalg utilities
    MLIRSCFDialect
    MLIRSCFTransforms  # SCF transformations
    MLIRSCFUtils  # SCF utilities
    MLIRMemRefDialect
    MLIRMemRefTransforms  # Memory optimizations
    MLIRMathDialect  # Math operations
    MLIRComplexDialect  # Complex numbers

    # Advanced Features for FEM
    MLIRVectorDialect
    MLIRVectorTransforms  # SIMD optimizations
    MLIRVectorUtils
    MLIRSparseTensorDialect  # Critical for FEM
    MLIRSparseTensorTransforms
    MLIRSparseTensorUtils
    MLIRBufferizationDialect
    MLIRBufferizationTransforms

    # GPU Support (for future Metal potential)
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRGPUUtils
    MLIRAsyncDialect  # Async execution
    MLIRAsyncTransforms

    # Pattern & Transform Infrastructure
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRTransformDialect
    MLIRTransformDialectUtils

    # ALL Conversion Passes
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMathToLLVM
    MLIRTensorToLinalg
    MLIRLinalgToStandard
    MLIRVectorToSCF
    MLIRVectorToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRAsyncToLLVM
    MLIRComplexToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRControlFlowToLLVM

    # Execution Engine & JIT
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
    MLIRJitRunner
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRLLVMCommonConversion

    # LLVM Backend (comprehensive)
    LLVMCore
    LLVMSupport
    LLVMTarget
    LLVMAnalysis
    LLVMTransformUtils
    LLVMScalarOpts
    LLVMVectorize
    LLVMPasses
    LLVMExecutionEngine
    LLVMMCJIT
    LLVMOrcJIT

    # Architecture-specific (M4 Mac)
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Desc
    LLVMAArch64Info
    LLVMAArch64Utils
)

# Link advanced module with MAXIMUM comprehensive libraries
target_link_libraries(firedrake_mlir_advanced PRIVATE
    # Core Infrastructure (ALL available)
    MLIRIR
    MLIRSupport
    MLIRAnalysis
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRRewrite

    # ALL Essential Dialects with utilities
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRArithDialect
    MLIRArithUtils
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRMathDialect
    MLIRComplexDialect

    # Advanced Dialects (ALL)
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorUtils
    MLIRSparseTensorDialect
    MLIRSparseTensorTransforms
    MLIRSparseTensorUtils
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRGPUUtils
    MLIRAsyncDialect
    MLIRAsyncTransforms

    # Pattern & Transform Infrastructure
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRTransformDialect
    MLIRTransformDialectUtils

    # ALL Conversion Passes
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMathToLLVM
    MLIRTensorToLinalg
    MLIRLinalgToStandard
    MLIRVectorToSCF
    MLIRVectorToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRAsyncToLLVM
    MLIRComplexToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRControlFlowToLLVM

    # Execution Engine & JIT
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
    MLIRJitRunner
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRLLVMCommonConversion

    # LLVM Backend (comprehensive)
    LLVMCore
    LLVMSupport
    LLVMTarget
    LLVMAnalysis
    LLVMTransformUtils
    LLVMScalarOpts
    LLVMVectorize
    LLVMPasses
    LLVMExecutionEngine
    LLVMMCJIT
    LLVMOrcJIT

    # Architecture-specific (M4 Mac)
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Desc
    LLVMAArch64Info
    LLVMAArch64Utils
)

# Set output directory
set_target_properties(firedrake_mlir_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(firedrake_mlir_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(firedrake_mlir_direct PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(firedrake_mlir_advanced PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install targets
install(TARGETS firedrake_mlir_ext firedrake_mlir_native firedrake_mlir_direct
    LIBRARY DESTINATION ${Python_SITELIB}/firedrake/mlir_backend
)

#===----------------------------------------------------------------------===//
# Testing Configuration
#===----------------------------------------------------------------------===//

# Option to build tests
option(BUILD_TESTING "Build the comprehensive test suite" ON)

# Enable testing
if(BUILD_TESTING)
    enable_testing()

    # Add test subdirectory
    add_subdirectory(test)

    message(STATUS "Test suite enabled - run 'make test' to execute")
    message(STATUS "  Unit tests: make test_unit")
    message(STATUS "  Integration tests: make test_integration")
    message(STATUS "  All tests: make test_all")
endif()
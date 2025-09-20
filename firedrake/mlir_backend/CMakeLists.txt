cmake_minimum_required(VERSION 3.20)
project(FiredrakeMLIR)

# CRITICAL: Force shared library usage to avoid duplicate LLVM registrations
set(LLVM_USE_STATIC_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "" FORCE)
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "" FORCE)
set(MLIR_LINK_MLIR_DYLIB ON CACHE BOOL "" FORCE)

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
include_directories(${PROJECT_SOURCE_DIR}/include)  # For C API header
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})  # For generated .inc files

# Add definitions
add_definitions(${LLVM_DEFINITIONS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default visibility to hidden (only export what we explicitly mark)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

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

# Option to enforce single LLVM/MLIR owner pattern
option(FD_ENFORCE_SINGLE_OWNER "Disallow MLIR/LLVM link in py modules" ON)

# Function to verify Python modules don't link LLVM/MLIR directly
function(assert_no_llvm_mlir target)
    if(NOT FD_ENFORCE_SINGLE_OWNER)
        return()
    endif()

    get_target_property(libs ${target} LINK_LIBRARIES)
    if(libs)
        foreach(L IN LISTS libs)
            if(L MATCHES "(^|;)MLIR|(^|;)LLVM")
                if(NOT L STREQUAL "firedrake_mlir")
                    message(FATAL_ERROR
                        "Target ${target} illegally links ${L}. "
                        "Only 'firedrake_mlir' may link LLVM/MLIR. Use the C API.")
                endif()
            endif()
        endforeach()
    endif()
endfunction()

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Main Python module for MLIR backend
pybind11_add_module(firedrake_mlir_backend
    src/py/mlir_backend.cpp  # Consolidated Python interface
)

# Direct UFL to MLIR translator (bypasses GEM/Impero/Loopy)
pybind11_add_module(firedrake_mlir_ufl2mlir
    src/UFL2MLIR.cpp     # Clean, focused UFL→MLIR translation
)

# Complete UFL to MLIR translator with full assembly features
pybind11_add_module(firedrake_mlir_ufl2mlir_proper
    src/UFL2MLIRProper.cpp  # Complete implementation without simplification
)

# Link UFL2MLIR against our main library
target_link_libraries(firedrake_mlir_ufl2mlir PRIVATE
    firedrake_mlir  # Our main library already has all MLIR functionality
)
assert_no_llvm_mlir(firedrake_mlir_ufl2mlir)

# Link the proper implementation
target_link_libraries(firedrake_mlir_ufl2mlir_proper PRIVATE
    firedrake_mlir  # Our main library already has all MLIR functionality
)
assert_no_llvm_mlir(firedrake_mlir_ufl2mlir_proper)

# Core MLIR library - working implementation
add_library(firedrake_mlir SHARED
    src/CAPI.cpp                   # C API implementation
    src/FEMDialectSimple.cpp       # FEM dialect implementation (working version)
    src/GEMDialectSimple.cpp       # GEM dialect implementation (working version)
    src/FEMPatterns.cpp            # FEM patterns and transformations
    src/OptimizationPatterns.cpp   # Additional optimization patterns
    src/MLIRCompiler.cpp           # MLIR compilation infrastructure
    src/SparseMatrix.cpp           # Sparse matrix operations
    src/SparseTensorSupport.cpp    # Sparse tensor support
)

# FEM Assembly library with CSR operations
add_library(fem_assembly SHARED
    src/FEMAssemblyKernel.cpp
    src/SparseCSROperations.cpp
    src/BoundaryConditions.cpp
)

# Add dependencies on TableGen output
add_dependencies(firedrake_mlir FEMOpsIncGen GEMOpsIncGen)
add_dependencies(firedrake_mlir_backend FEMOpsIncGen GEMOpsIncGen)
add_dependencies(firedrake_mlir_ufl2mlir FEMOpsIncGen GEMOpsIncGen)

# Link Python modules against our main library
target_link_libraries(firedrake_mlir_backend PRIVATE
    firedrake_mlir  # Our main library already has all MLIR functionality
)
assert_no_llvm_mlir(firedrake_mlir_backend)

# Link ONLY shared LLVM/MLIR libraries to avoid duplicate registrations
target_link_libraries(firedrake_mlir PRIVATE
    LLVM                        # Single shared libLLVM.dylib
    MLIR                        # Single shared libMLIR.dylib
    MLIRExecutionEngine         # JIT execution engine (correct target name)
)

# Link FEM assembly library to shared libraries
target_link_libraries(fem_assembly PRIVATE
    LLVM                        # Single shared libLLVM.dylib
    MLIR                        # Single shared libMLIR.dylib
    MLIRExecutionEngine         # JIT execution engine (correct target name)
)

# Set output directory for Python modules
set_target_properties(firedrake_mlir_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/build"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(firedrake_mlir_ufl2mlir PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/build"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(firedrake_mlir_ufl2mlir_proper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/build"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(firedrake_mlir PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    MACOSX_RPATH ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@loader_path;/Users/jerry/llvm-install/lib"
)

set_target_properties(fem_assembly PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

# Install targets
install(TARGETS firedrake_mlir_backend firedrake_mlir_ufl2mlir firedrake_mlir fem_assembly
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
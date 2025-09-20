/*
 * firedrake_mlir_c.h - C API for Firedrake MLIR backend
 *
 * This header defines the ONLY interface between the core library
 * (which contains all MLIR/LLVM code) and Python extension modules.
 *
 * CRITICAL: No MLIR/LLVM types are exposed. Only opaque handles.
 */

#pragma once

#include <stdint.h>  // For int64_t

#ifdef __cplusplus
extern "C" {
#endif

// Export macro for public API functions
#if defined(_WIN32)
  #define FD_API __declspec(dllexport)
#else
  #define FD_API __attribute__((visibility("default")))
#endif

// Opaque handle types (no MLIR types exposed)
typedef void* FdCompiler;
typedef void* FdModule;
typedef void* FdKernel;
typedef void* FdAssembly;

// Error handling
typedef struct {
    int code;
    char* message;  // Caller must free with fd_free_string
} FdError;

//===----------------------------------------------------------------------===//
// Core initialization (must be called once before any other API)
//===----------------------------------------------------------------------===//

// Initialize LLVM targets and MLIR dialects (idempotent)
// Returns 0 on success, non-zero on failure
FD_API int fd_init_once(void);

//===----------------------------------------------------------------------===//
// Compiler API
//===----------------------------------------------------------------------===//

// Create a new MLIR compiler instance
FD_API FdCompiler fd_compiler_create(void);

// Destroy a compiler instance
FD_API void fd_compiler_destroy(FdCompiler compiler);

// Generate FEM assembly kernel
// Returns 0 on success, fills error on failure
FD_API int fd_compiler_generate_fem_assembly(
    FdCompiler compiler,
    int num_elements,
    int dofs_per_element,
    int quad_points,
    FdError* error
);

// Verify the generated MLIR module
FD_API int fd_compiler_verify(FdCompiler compiler);

// Apply optimization passes
// level: 0=none, 1=basic, 2=standard, 3=aggressive
FD_API int fd_compiler_optimize(FdCompiler compiler, int level);

// Lower MLIR to LLVM IR
FD_API int fd_compiler_lower_to_llvm(FdCompiler compiler);

// Create JIT execution engine
FD_API int fd_compiler_create_jit(FdCompiler compiler, FdKernel* out_kernel);

// Get MLIR text representation (for debugging)
// IMPORTANT: Caller must free returned string with fd_free()
FD_API char* fd_compiler_get_mlir_text(FdCompiler compiler);

// Get LLVM IR text representation (for debugging)
// IMPORTANT: Caller must free returned string with fd_free()
FD_API char* fd_compiler_get_llvm_text(FdCompiler compiler);

//===----------------------------------------------------------------------===//
// UFL to MLIR compilation
//===----------------------------------------------------------------------===//

// Compile UFL expression directly to MLIR
// Returns 0 on success, fills error on failure
// out_mlir_text: if non-null, receives MLIR text (caller frees with fd_free_string)
FD_API int fd_compile_ufl(
    FdCompiler compiler,
    const char* ufl_expr,
    const char* params_json,
    char** out_mlir_text,
    FdError* error
);

//===----------------------------------------------------------------------===//
// MLIR Builder API - Expose full MLIR building capabilities
//===----------------------------------------------------------------------===//

// Opaque handles for MLIR building
typedef void* FdBuilder;
typedef void* FdValue;
typedef void* FdType;
typedef void* FdFunction;
typedef void* FdBlock;

// Create an OpBuilder for the compiler's module
FD_API FdBuilder fd_compiler_create_builder(FdCompiler compiler);

// Destroy a builder
FD_API void fd_builder_destroy(FdBuilder builder);

// Get the module from compiler (for builder insertion)
FD_API FdModule fd_compiler_get_module(FdCompiler compiler);

// Create types
FD_API FdType fd_builder_get_f64_type(FdBuilder builder);
FD_API FdType fd_builder_get_index_type(FdBuilder builder);
FD_API FdType fd_builder_get_memref_type(FdBuilder builder,
                                        int rank,
                                        const int64_t* shape,  // -1 for dynamic
                                        FdType element_type);

// Create function
FD_API FdFunction fd_builder_create_function(FdBuilder builder,
                                            const char* name,
                                            int num_args,
                                            FdType* arg_types);

// Add function to module
FD_API void fd_module_add_function(FdModule module, FdFunction func);

// Get function entry block
FD_API FdBlock fd_function_get_entry_block(FdFunction func);

// Get block arguments
FD_API int fd_block_get_num_arguments(FdBlock block);
FD_API FdValue fd_block_get_argument(FdBlock block, int index);

// Set insertion point
FD_API void fd_builder_set_insertion_point(FdBuilder builder, FdBlock block);

// Create constants
FD_API FdValue fd_builder_create_constant_index(FdBuilder builder, int64_t value);
FD_API FdValue fd_builder_create_constant_f64(FdBuilder builder, double value);

// Create loops
FD_API FdValue fd_builder_create_scf_for(FdBuilder builder,
                                        FdValue lower_bound,
                                        FdValue upper_bound,
                                        FdValue step,
                                        FdBlock* loop_body);

// Memory operations
FD_API FdValue fd_builder_create_memref_alloca(FdBuilder builder,
                                              FdType memref_type);
FD_API void fd_builder_create_memref_store(FdBuilder builder,
                                          FdValue value,
                                          FdValue memref,
                                          int num_indices,
                                          FdValue* indices);
FD_API FdValue fd_builder_create_memref_load(FdBuilder builder,
                                            FdValue memref,
                                            int num_indices,
                                            FdValue* indices);
FD_API void fd_builder_create_memref_dealloc(FdBuilder builder, FdValue memref);

// Arithmetic operations
FD_API FdValue fd_builder_create_addf(FdBuilder builder, FdValue lhs, FdValue rhs);
FD_API FdValue fd_builder_create_mulf(FdBuilder builder, FdValue lhs, FdValue rhs);
FD_API FdValue fd_builder_create_addi(FdBuilder builder, FdValue lhs, FdValue rhs);
FD_API FdValue fd_builder_create_muli(FdBuilder builder, FdValue lhs, FdValue rhs);

// Control flow
FD_API void fd_builder_create_scf_yield(FdBuilder builder);
FD_API void fd_builder_create_func_return(FdBuilder builder);

//===----------------------------------------------------------------------===//
// FEM Assembly API
//===----------------------------------------------------------------------===//

// Create FEM assembly context
FD_API FdAssembly fd_assembly_create(void);

// Destroy FEM assembly context
FD_API void fd_assembly_destroy(FdAssembly assembly);

// Compile FEM kernel for given polynomial degree
// Returns kernel handle on success, NULL on failure
FD_API FdKernel fd_assembly_compile_kernel(FdAssembly assembly, int degree);

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

// Free any memory allocated by the API (strings, errors, etc.)
FD_API void fd_free(void* p);

//===----------------------------------------------------------------------===//
// Dialect registration (for advanced users)
//===----------------------------------------------------------------------===//

// Register a custom dialect (advanced use only)
FD_API int fd_register_dialect(const char* dialect_name);

// Check if a dialect is registered
FD_API int fd_is_dialect_registered(const char* dialect_name);

#ifdef __cplusplus
}
#endif
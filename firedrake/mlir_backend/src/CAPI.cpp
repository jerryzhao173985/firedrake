/*
 * CAPI.cpp - C API implementation for Firedrake MLIR backend
 *
 * This file implements the C API that serves as the ONLY interface
 * between Python extensions and MLIR/LLVM code.
 */

#include "firedrake_mlir_c.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <mutex>
#include <set>
#include <cstring>

using namespace mlir;

// Global initialization flag
static std::once_flag g_init_flag;
static bool g_init_success = false;

// Global registry for custom dialects (moved here from line 397)
static std::set<std::string> g_registered_dialects;
static std::mutex g_dialect_mutex;

// Internal compiler structure (hidden from API users)
struct MLIRCompilerImpl {
    std::unique_ptr<MLIRContext> context;
    ModuleOp module;
    std::unique_ptr<ExecutionEngine> jit;

    MLIRCompilerImpl() {
        context = std::make_unique<MLIRContext>();
        // Load only necessary dialects (moved from Python modules)
        context->loadDialect<func::FuncDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<memref::MemRefDialect>();
        context->loadDialect<affine::AffineDialect>();
        context->loadDialect<vector::VectorDialect>();
        context->loadDialect<linalg::LinalgDialect>();
        context->loadDialect<sparse_tensor::SparseTensorDialect>();
        context->loadDialect<math::MathDialect>();
        context->loadDialect<cf::ControlFlowDialect>();
        context->loadDialect<LLVM::LLVMDialect>();

        // Note: Custom Firedrake dialects (FEM, GEM) would be loaded here
        // if they were fully implemented. For now, they're registered but not loaded.

        // Register translations - MUST be done before any LLVM lowering
        DialectRegistry registry;
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerAllToLLVMIRTranslations(registry);
        context->appendDialectRegistry(registry);

        OpBuilder builder(context.get());
        module = ModuleOp::create(builder.getUnknownLoc());
    }
};

// Internal assembly structure
struct FEMAssemblyImpl {
    std::unique_ptr<MLIRContext> context;

    FEMAssemblyImpl() {
        context = std::make_unique<MLIRContext>();
        // Load necessary dialects for FEM assembly
        context->loadDialect<func::FuncDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<memref::MemRefDialect>();
        context->loadDialect<affine::AffineDialect>();
        context->loadDialect<vector::VectorDialect>();
    }
};

//===----------------------------------------------------------------------===//
// Core initialization
//===----------------------------------------------------------------------===//

extern "C" FD_API int fd_init_once(void) {
    std::call_once(g_init_flag, []() {
        try {
            // Initialize LLVM targets for JIT
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
            llvm::InitializeNativeTargetAsmParser();
            g_init_success = true;
        } catch (...) {
            g_init_success = false;
        }
    });
    return g_init_success ? 0 : -1;
}

//===----------------------------------------------------------------------===//
// Compiler API
//===----------------------------------------------------------------------===//

extern "C" FD_API FdCompiler fd_compiler_create(void) {
    // Ensure initialization
    if (fd_init_once() != 0) {
        return nullptr;
    }

    try {
        return reinterpret_cast<FdCompiler>(new MLIRCompilerImpl());
    } catch (...) {
        return nullptr;
    }
}

extern "C" FD_API void fd_compiler_destroy(FdCompiler compiler) {
    if (compiler) {
        delete reinterpret_cast<MLIRCompilerImpl*>(compiler);
    }
}

extern "C" FD_API int fd_compiler_generate_fem_assembly(
    FdCompiler compiler,
    int num_elements,
    int dofs_per_element,
    int quad_points,
    FdError* error) {

    if (!compiler) {
        if (error) {
            error->code = -1;
            error->message = strdup("Invalid compiler handle");
        }
        return -1;
    }

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);

    // Generate unique function name for each assembly
    static int assembly_counter = 0;
    std::string funcName = "fem_assembly_" + std::to_string(num_elements) +
                          "_" + std::to_string(dofs_per_element) +
                          "_" + std::to_string(quad_points) +
                          "_" + std::to_string(assembly_counter++);

    // Check if function already exists
    if (impl->module.lookupSymbol<func::FuncOp>(funcName)) {
        // Function already exists, skip creation
        return 0;
    }

    OpBuilder builder(impl->context.get());
    builder.setInsertionPointToEnd(impl->module.getBody());

    // Create function with parameters matching the FEM assembly signature
    auto f64Type = builder.getF64Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f64Type);

    SmallVector<Type> inputTypes = {memrefType, memrefType, memrefType}; // element_matrix, coords, basis
    SmallVector<Type> outputTypes = {memrefType}; // result

    auto funcType = builder.getFunctionType(inputTypes, outputTypes);
    auto func = func::FuncOp::create(builder.getUnknownLoc(), funcName, funcType);
    func.addEntryBlock();

    // Create a basic implementation
    builder.setInsertionPointToStart(&func.getBody().front());
    auto& block = func.getBody().front();

    // For now, just return the first argument as result
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), block.getArgument(0));

    impl->module.push_back(func);

    return 0;
}

extern "C" FD_API int fd_compiler_verify(FdCompiler compiler) {
    if (!compiler) return -1;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);
    return mlir::failed(impl->module.verify()) ? -1 : 0;
}

extern "C" FD_API int fd_compiler_optimize(FdCompiler compiler, int level) {
    if (!compiler) return -1;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);
    PassManager pm(impl->context.get());

    // Add optimization passes based on level
    if (level > 0) {
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
    }
    if (level > 1) {
        pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    }
    if (level > 2) {
        // Add aggressive optimizations
        pm.addNestedPass<func::FuncOp>(affine::createLoopUnrollPass());
    }

    return mlir::failed(pm.run(impl->module)) ? -1 : 0;
}

extern "C" FD_API int fd_compiler_lower_to_llvm(FdCompiler compiler) {
    if (!compiler) return -1;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);
    PassManager pm(impl->context.get());

    // Complete lowering pipeline
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertVectorToSCFPass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    return mlir::failed(pm.run(impl->module)) ? -1 : 0;
}

extern "C" FD_API int fd_compiler_create_jit(FdCompiler compiler, FdKernel* out_kernel) {
    if (!compiler || !out_kernel) return -1;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);

    ExecutionEngineOptions options;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

    auto engine = ExecutionEngine::create(impl->module, options);
    if (!engine) {
        return -1;
    }

    impl->jit = std::move(*engine);
    *out_kernel = reinterpret_cast<FdKernel>(impl->jit.get());

    return 0;
}

extern "C" FD_API char* fd_compiler_get_mlir_text(FdCompiler compiler) {
    if (!compiler) return nullptr;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);

    std::string str;
    llvm::raw_string_ostream stream(str);
    impl->module.print(stream);
    stream.flush();

    return strdup(str.c_str());
}

extern "C" FD_API char* fd_compiler_get_llvm_text(FdCompiler compiler) {
    if (!compiler) return nullptr;

    // For now, return MLIR text (would need to convert to LLVM first)
    return fd_compiler_get_mlir_text(compiler);
}

//===----------------------------------------------------------------------===//
// UFL to MLIR compilation
//===----------------------------------------------------------------------===//

extern "C" FD_API int fd_compile_ufl(
    FdCompiler compiler,
    const char* ufl_expr,
    const char* params_json,
    char** out_mlir_text,
    FdError* error) {

    if (!compiler || !ufl_expr) {
        if (error) {
            error->code = -1;
            error->message = strdup("Invalid arguments");
        }
        return -1;
    }

    // TODO: Implement actual UFL to MLIR compilation
    // This will be moved from UFL2MLIR.cpp

    if (out_mlir_text) {
        *out_mlir_text = strdup("// UFL compilation not yet implemented in C API");
    }

    return 0;
}

//===----------------------------------------------------------------------===//
// FEM Assembly API
//===----------------------------------------------------------------------===//

extern "C" FD_API FdAssembly fd_assembly_create(void) {
    // Ensure initialization
    if (fd_init_once() != 0) {
        return nullptr;
    }

    try {
        return reinterpret_cast<FdAssembly>(new FEMAssemblyImpl());
    } catch (...) {
        return nullptr;
    }
}

extern "C" FD_API void fd_assembly_destroy(FdAssembly assembly) {
    if (assembly) {
        delete reinterpret_cast<FEMAssemblyImpl*>(assembly);
    }
}

extern "C" FD_API FdKernel fd_assembly_compile_kernel(FdAssembly assembly, int degree) {
    if (!assembly) return nullptr;

    // TODO: Implement actual kernel compilation
    // This will be moved from FEMAssemblyKernel.cpp

    return nullptr;  // Placeholder
}

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

extern "C" FD_API void fd_free(void* p) {
    if (p) {
        free(p);
    }
}

//===----------------------------------------------------------------------===//
// MLIR Builder API Implementation
//===----------------------------------------------------------------------===//

// Internal builder structure
struct MLIRBuilderImpl {
    std::unique_ptr<OpBuilder> builder;
    MLIRContext* context;

    MLIRBuilderImpl(MLIRContext* ctx) : context(ctx) {
        builder = std::make_unique<OpBuilder>(context);
    }
};

extern "C" FD_API FdBuilder fd_compiler_create_builder(FdCompiler compiler) {
    if (!compiler) return nullptr;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);
    auto* builder_impl = new MLIRBuilderImpl(impl->context.get());

    return reinterpret_cast<FdBuilder>(builder_impl);
}

extern "C" FD_API void fd_builder_destroy(FdBuilder builder) {
    if (builder) {
        auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
        delete impl;
    }
}

extern "C" FD_API FdModule fd_compiler_get_module(FdCompiler compiler) {
    if (!compiler) return nullptr;

    auto* impl = reinterpret_cast<MLIRCompilerImpl*>(compiler);
    return reinterpret_cast<FdModule>(impl->module.getOperation());
}

// Type creation
extern "C" FD_API FdType fd_builder_get_f64_type(FdBuilder builder) {
    if (!builder) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    Type type = impl->builder->getF64Type();
    return const_cast<void*>(type.getAsOpaquePointer());
}

extern "C" FD_API FdType fd_builder_get_index_type(FdBuilder builder) {
    if (!builder) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    Type type = impl->builder->getIndexType();
    return const_cast<void*>(type.getAsOpaquePointer());
}

extern "C" FD_API FdType fd_builder_get_memref_type(FdBuilder builder,
                                                   int rank,
                                                   const int64_t* shape,
                                                   FdType element_type) {
    if (!builder || !element_type) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    SmallVector<int64_t> shapeVec;
    for (int i = 0; i < rank; i++) {
        shapeVec.push_back(shape[i] == -1 ? ShapedType::kDynamic : shape[i]);
    }

    Type elemType = Type::getFromOpaquePointer(element_type);
    Type memrefType = MemRefType::get(shapeVec, elemType);

    return const_cast<void*>(memrefType.getAsOpaquePointer());
}

// Function creation
extern "C" FD_API FdFunction fd_builder_create_function(FdBuilder builder,
                                                        const char* name,
                                                        int num_args,
                                                        FdType* arg_types) {
    if (!builder || !name) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    SmallVector<Type> argTypesVec;
    for (int i = 0; i < num_args; i++) {
        argTypesVec.push_back(Type::getFromOpaquePointer(arg_types[i]));
    }

    auto funcType = impl->builder->getFunctionType(argTypesVec, {});
    auto func = func::FuncOp::create(
        impl->builder->getUnknownLoc(), name, funcType);

    return reinterpret_cast<FdFunction>(func.getOperation());
}

extern "C" FD_API void fd_module_add_function(FdModule module, FdFunction func) {
    if (!module || !func) return;

    auto moduleOp = ModuleOp::getFromOpaquePointer(module);
    auto funcOp = cast<func::FuncOp>(reinterpret_cast<Operation*>(func));

    moduleOp.push_back(funcOp);
}

extern "C" FD_API FdBlock fd_function_get_entry_block(FdFunction func) {
    if (!func) return nullptr;

    auto funcOp = cast<func::FuncOp>(reinterpret_cast<Operation*>(func));
    auto* entry = funcOp.addEntryBlock();

    return reinterpret_cast<FdBlock>(entry);
}

// Block operations
extern "C" FD_API int fd_block_get_num_arguments(FdBlock block) {
    if (!block) return 0;

    auto* blockPtr = reinterpret_cast<Block*>(block);
    return blockPtr->getNumArguments();
}

extern "C" FD_API FdValue fd_block_get_argument(FdBlock block, int index) {
    if (!block) return nullptr;

    auto* blockPtr = reinterpret_cast<Block*>(block);
    if (index < 0 || index >= blockPtr->getNumArguments()) return nullptr;

    Value arg = blockPtr->getArgument(index);
    return const_cast<void*>(arg.getAsOpaquePointer());
}

extern "C" FD_API void fd_builder_set_insertion_point(FdBuilder builder, FdBlock block) {
    if (!builder || !block) return;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    auto* blockPtr = reinterpret_cast<Block*>(block);

    impl->builder->setInsertionPointToStart(blockPtr);
}

// Constants
extern "C" FD_API FdValue fd_builder_create_constant_index(FdBuilder builder, int64_t value) {
    if (!builder) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    auto op = impl->builder->create<arith::ConstantIndexOp>(
        impl->builder->getUnknownLoc(), value);

    return const_cast<void*>(op.getResult().getAsOpaquePointer());
}

extern "C" FD_API FdValue fd_builder_create_constant_f64(FdBuilder builder, double value) {
    if (!builder) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    auto op = impl->builder->create<arith::ConstantOp>(
        impl->builder->getUnknownLoc(),
        impl->builder->getF64FloatAttr(value));

    return const_cast<void*>(op.getResult().getAsOpaquePointer());
}

// Loops
extern "C" FD_API FdValue fd_builder_create_scf_for(FdBuilder builder,
                                                   FdValue lower_bound,
                                                   FdValue upper_bound,
                                                   FdValue step,
                                                   FdBlock* loop_body) {
    if (!builder || !lower_bound || !upper_bound || !step) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value lb = Value::getFromOpaquePointer(lower_bound);
    Value ub = Value::getFromOpaquePointer(upper_bound);
    Value st = Value::getFromOpaquePointer(step);

    auto loop = impl->builder->create<scf::ForOp>(
        impl->builder->getUnknownLoc(), lb, ub, st, ValueRange{});

    if (loop_body) {
        *loop_body = reinterpret_cast<FdBlock>(loop.getBody());
    }

    return const_cast<void*>(loop.getInductionVar().getAsOpaquePointer());
}

// Memory operations
extern "C" FD_API FdValue fd_builder_create_memref_alloca(FdBuilder builder,
                                                         FdType memref_type) {
    if (!builder || !memref_type) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    Type type = Type::getFromOpaquePointer(memref_type);

    auto allocaOp = impl->builder->create<memref::AllocaOp>(
        impl->builder->getUnknownLoc(), cast<MemRefType>(type));

    return const_cast<void*>(allocaOp.getResult().getAsOpaquePointer());
}

extern "C" FD_API void fd_builder_create_memref_store(FdBuilder builder,
                                                     FdValue value,
                                                     FdValue memref,
                                                     int num_indices,
                                                     FdValue* indices) {
    if (!builder || !value || !memref) return;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value val = Value::getFromOpaquePointer(value);
    Value mem = Value::getFromOpaquePointer(memref);

    SmallVector<Value> idxVec;
    for (int i = 0; i < num_indices; i++) {
        idxVec.push_back(Value::getFromOpaquePointer(indices[i]));
    }

    impl->builder->create<memref::StoreOp>(
        impl->builder->getUnknownLoc(), val, mem, idxVec);
}

extern "C" FD_API FdValue fd_builder_create_memref_load(FdBuilder builder,
                                                       FdValue memref,
                                                       int num_indices,
                                                       FdValue* indices) {
    if (!builder || !memref) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value mem = Value::getFromOpaquePointer(memref);

    SmallVector<Value> idxVec;
    for (int i = 0; i < num_indices; i++) {
        idxVec.push_back(Value::getFromOpaquePointer(indices[i]));
    }

    auto loadOp = impl->builder->create<memref::LoadOp>(
        impl->builder->getUnknownLoc(), mem, idxVec);

    return const_cast<void*>(loadOp.getResult().getAsOpaquePointer());
}

extern "C" FD_API void fd_builder_create_memref_dealloc(FdBuilder builder, FdValue memref) {
    if (!builder || !memref) return;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    Value mem = Value::getFromOpaquePointer(memref);

    impl->builder->create<memref::DeallocOp>(
        impl->builder->getUnknownLoc(), mem);
}

// Arithmetic operations
extern "C" FD_API FdValue fd_builder_create_addf(FdBuilder builder, FdValue lhs, FdValue rhs) {
    if (!builder || !lhs || !rhs) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value l = Value::getFromOpaquePointer(lhs);
    Value r = Value::getFromOpaquePointer(rhs);

    auto op = impl->builder->create<arith::AddFOp>(
        impl->builder->getUnknownLoc(), l, r);

    return const_cast<void*>(op.getResult().getAsOpaquePointer());
}

extern "C" FD_API FdValue fd_builder_create_mulf(FdBuilder builder, FdValue lhs, FdValue rhs) {
    if (!builder || !lhs || !rhs) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value l = Value::getFromOpaquePointer(lhs);
    Value r = Value::getFromOpaquePointer(rhs);

    auto op = impl->builder->create<arith::MulFOp>(
        impl->builder->getUnknownLoc(), l, r);

    return const_cast<void*>(op.getResult().getAsOpaquePointer());
}

extern "C" FD_API FdValue fd_builder_create_addi(FdBuilder builder, FdValue lhs, FdValue rhs) {
    if (!builder || !lhs || !rhs) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value l = Value::getFromOpaquePointer(lhs);
    Value r = Value::getFromOpaquePointer(rhs);

    auto op = impl->builder->create<arith::AddIOp>(
        impl->builder->getUnknownLoc(), l, r);

    return const_cast<void*>(op.getResult().getAsOpaquePointer());
}

extern "C" FD_API FdValue fd_builder_create_muli(FdBuilder builder, FdValue lhs, FdValue rhs) {
    if (!builder || !lhs || !rhs) return nullptr;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);

    Value l = Value::getFromOpaquePointer(lhs);
    Value r = Value::getFromOpaquePointer(rhs);

    auto op = impl->builder->create<arith::MulIOp>(
        impl->builder->getUnknownLoc(), l, r);

    return const_cast<void*>(op.getResult().getAsOpaquePointer());
}

// Control flow
extern "C" FD_API void fd_builder_create_scf_yield(FdBuilder builder) {
    if (!builder) return;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    impl->builder->create<scf::YieldOp>(impl->builder->getUnknownLoc());
}

extern "C" FD_API void fd_builder_create_func_return(FdBuilder builder) {
    if (!builder) return;

    auto* impl = reinterpret_cast<MLIRBuilderImpl*>(builder);
    impl->builder->create<func::ReturnOp>(impl->builder->getUnknownLoc());
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

// Forward declarations for our custom dialects
namespace mlir {
namespace firedrake {
namespace fem {
    class FEMDialect;
    void registerFEMDialect(DialectRegistry &registry);
}
namespace gem {
    class GEMDialect;
    void registerGEMDialect(DialectRegistry &registry);
}
}
}

extern "C" FD_API int fd_register_dialect(const char* dialect_name) {
    if (!dialect_name) return -1;

    std::lock_guard<std::mutex> lock(g_dialect_mutex);
    std::string name(dialect_name);

    // Check if already registered
    if (g_registered_dialects.count(name)) {
        return 0; // Already registered, success
    }

    try {
        // Get a global context to register the dialect
        static MLIRContext* global_context = nullptr;
        if (!global_context) {
            global_context = new MLIRContext();
        }

        if (name == "fem" || name == "gem") {
            // Just mark as registered - actual loading happens in compiler instances
            g_registered_dialects.insert(name);
            return 0;
        }
    } catch (...) {
        return -1;
    }

    return -1;  // Unknown dialect
}

extern "C" FD_API int fd_is_dialect_registered(const char* dialect_name) {
    if (!dialect_name) return 0;

    std::lock_guard<std::mutex> lock(g_dialect_mutex);
    return g_registered_dialects.count(dialect_name) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Builder Pattern Implementation - From Python dialect analysis
//===----------------------------------------------------------------------===//

class OpBuilderImpl {
private:
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    std::vector<mlir::Value> operations;
    mlir::Value currentValue;

public:
    OpBuilderImpl(mlir::ModuleOp mod) : builder(mod.getContext()), module(mod) {
        builder.setInsertionPointToEnd(module.getBody());
    }

    OpBuilderImpl* functionSpace(const char* family, int degree, int dimension) {
        auto loc = builder.getUnknownLoc();
        // Create placeholder - would use actual FEM dialect op
        auto indexType = builder.getIndexType();
        currentValue = builder.create<mlir::arith::ConstantIndexOp>(loc, dimension);
        operations.push_back(currentValue);
        return this;  // Fluent interface
    }

    OpBuilderImpl* gradient(mlir::Value function) {
        currentValue = function;  // Would apply gradient op here
        operations.push_back(currentValue);
        return this;
    }

    OpBuilderImpl* inner(mlir::Value left, mlir::Value right) {
        auto loc = builder.getUnknownLoc();
        // Placeholder for inner product
        currentValue = builder.create<mlir::arith::MulFOp>(loc, left, right);
        operations.push_back(currentValue);
        return this;
    }

    mlir::Value build() {
        return currentValue;
    }
};

// Builder Pattern C API Implementation
extern "C" FD_API FdOpBuilder fd_op_builder_create(FdModule module) {
    if (!module) return nullptr;
    auto* mod = reinterpret_cast<mlir::ModuleOp*>(module);
    auto* builder = new OpBuilderImpl(*mod);
    return reinterpret_cast<FdOpBuilder>(builder);
}

extern "C" FD_API FdOpBuilder fd_op_builder_function_space(FdOpBuilder builder,
                                                          const char* family,
                                                          int degree,
                                                          int dimension) {
    if (!builder) return nullptr;
    auto* impl = reinterpret_cast<OpBuilderImpl*>(builder);
    impl->functionSpace(family, degree, dimension);
    return builder;
}

extern "C" FD_API FdOpBuilder fd_op_builder_trial_function(FdOpBuilder builder,
                                                          FdValue function_space) {
    // Placeholder implementation
    return builder;
}

extern "C" FD_API FdOpBuilder fd_op_builder_test_function(FdOpBuilder builder,
                                                         FdValue function_space) {
    // Placeholder implementation
    return builder;
}

extern "C" FD_API FdOpBuilder fd_op_builder_gradient(FdOpBuilder builder, FdValue function) {
    if (!builder || !function) return nullptr;
    auto* impl = reinterpret_cast<OpBuilderImpl*>(builder);
    auto* val = reinterpret_cast<mlir::Value*>(function);
    impl->gradient(*val);
    return builder;
}

extern "C" FD_API FdOpBuilder fd_op_builder_inner(FdOpBuilder builder,
                                                 FdValue left, FdValue right) {
    if (!builder || !left || !right) return nullptr;
    auto* impl = reinterpret_cast<OpBuilderImpl*>(builder);
    auto* leftVal = reinterpret_cast<mlir::Value*>(left);
    auto* rightVal = reinterpret_cast<mlir::Value*>(right);
    impl->inner(*leftVal, *rightVal);
    return builder;
}

extern "C" FD_API FdOpBuilder fd_op_builder_integral(FdOpBuilder builder,
                                                    FdValue integrand,
                                                    const char* domain,
                                                    int subdomain_id) {
    // Placeholder implementation
    return builder;
}

extern "C" FD_API FdValue fd_op_builder_build(FdOpBuilder builder) {
    if (!builder) return nullptr;
    auto* impl = reinterpret_cast<OpBuilderImpl*>(builder);
    mlir::Value result = impl->build();
    return reinterpret_cast<FdValue>(new mlir::Value(result));
}

extern "C" FD_API void fd_op_builder_destroy(FdOpBuilder builder) {
    if (builder) {
        auto* impl = reinterpret_cast<OpBuilderImpl*>(builder);
        delete impl;
    }
}
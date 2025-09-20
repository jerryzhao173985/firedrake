/*
 * MLIRCompiler.cpp - Core MLIR Compiler for Firedrake
 *
 * This is the CLEAN, WORKING implementation that replaces all the duplicates.
 * Direct UFL → MLIR → Optimized Code
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

// Transformation passes
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCContext.h"

#include <memory>
#include <vector>

namespace firedrake {
namespace mlir_backend {

using namespace mlir;

class MLIRCompiler {
private:
    std::unique_ptr<MLIRContext> context;
    std::unique_ptr<OpBuilder> builder;
    ModuleOp module;
    std::unique_ptr<ExecutionEngine> executionEngine;

public:
    MLIRCompiler() {
        // CRITICAL: Initialize LLVM targets for JIT compilation
        static bool llvmInitialized = false;
        if (!llvmInitialized) {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
            llvm::InitializeNativeTargetAsmParser();
            llvmInitialized = true;
        }

        // Initialize MLIR context with necessary dialects
        context = std::make_unique<MLIRContext>();
        context->loadDialect<affine::AffineDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<func::FuncDialect>();
        context->loadDialect<linalg::LinalgDialect>();
        context->loadDialect<memref::MemRefDialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<sparse_tensor::SparseTensorDialect>();
        context->loadDialect<vector::VectorDialect>();

        // CRITICAL: Register ALL dialect translations for LLVM IR
        DialectRegistry registry;
        // Register builtin dialect translation first
        mlir::registerBuiltinDialectTranslation(registry);
        // Register all other translations
        mlir::registerAllToLLVMIRTranslations(registry);
        context->appendDialectRegistry(registry);

        builder = std::make_unique<OpBuilder>(context.get());
        module = ModuleOp::create(builder->getUnknownLoc());

        // Register translations in the module context after creation
        module.getContext()->appendDialectRegistry(registry);
    }

    //===------------------------------------------------------------------===//
    // Core FEM Assembly Generation
    //===------------------------------------------------------------------===//

    func::FuncOp generateFEMAssembly(
        int numElements, int dofsPerElement, int quadPoints) {

        // Clear module operations if any exist
        module.getBody()->clear();

        auto loc = builder->getUnknownLoc();
        auto f64Type = builder->getF64Type();
        auto indexType = builder->getIndexType();

        // Function signature for FEM assembly kernel
        // Args: global_matrix, element_matrices, connectivity, basis, weights
        SmallVector<Type> argTypes = {
            // Global stiffness matrix (sparse)
            MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f64Type),
            // Element matrices
            MemRefType::get({numElements, dofsPerElement, dofsPerElement}, f64Type),
            // Connectivity array
            MemRefType::get({numElements, dofsPerElement}, indexType),
            // Basis functions at quadrature points
            MemRefType::get({dofsPerElement, quadPoints}, f64Type),
            // Quadrature weights
            MemRefType::get({quadPoints}, f64Type)
        };

        auto funcType = builder->getFunctionType(argTypes, {});
        auto func = func::FuncOp::create(loc, "fem_assembly_kernel", funcType);

        // Create function body
        auto* entryBlock = func.addEntryBlock();
        builder->setInsertionPointToStart(entryBlock);

        // Get function arguments
        Value globalMatrix = entryBlock->getArgument(0);
        Value elementMatrices = entryBlock->getArgument(1);
        Value connectivity = entryBlock->getArgument(2);
        Value basis = entryBlock->getArgument(3);
        Value weights = entryBlock->getArgument(4);

        // Generate assembly loops
        generateAssemblyLoops(globalMatrix, elementMatrices, connectivity,
                            basis, weights, numElements, dofsPerElement, quadPoints);

        builder->create<func::ReturnOp>(loc);

        // Add function to module
        module.push_back(func);

        return func;
    }

private:
    void generateAssemblyLoops(
        Value globalMatrix, Value elementMatrices, Value connectivity,
        Value basis, Value weights, int numElements, int dofsPerElement, int quadPoints) {

        auto loc = builder->getUnknownLoc();

        // Constants
        auto c0 = builder->create<arith::ConstantIndexOp>(loc, 0);
        auto c1 = builder->create<arith::ConstantIndexOp>(loc, 1);
        auto cNumElements = builder->create<arith::ConstantIndexOp>(loc, numElements);
        auto cDofsPerElement = builder->create<arith::ConstantIndexOp>(loc, dofsPerElement);
        auto cQuadPoints = builder->create<arith::ConstantIndexOp>(loc, quadPoints);

        // Element loop
        auto elemLoop = builder->create<scf::ForOp>(loc, c0, cNumElements, c1);
        builder->setInsertionPointToStart(elemLoop.getBody());
        Value elemIdx = elemLoop.getInductionVar();

        // Test function loop
        auto testLoop = builder->create<scf::ForOp>(loc, c0, cDofsPerElement, c1);
        builder->setInsertionPointToStart(testLoop.getBody());
        Value i = testLoop.getInductionVar();

        // Trial function loop
        auto trialLoop = builder->create<scf::ForOp>(loc, c0, cDofsPerElement, c1);
        builder->setInsertionPointToStart(trialLoop.getBody());
        Value j = trialLoop.getInductionVar();

        // Initialize accumulator
        auto zero = builder->create<arith::ConstantOp>(
            loc, builder->getF64FloatAttr(0.0));

        // Quadrature loop with accumulation
        auto quadLoop = builder->create<scf::ForOp>(
            loc, c0, cQuadPoints, c1, ValueRange{zero});
        builder->setInsertionPointToStart(quadLoop.getBody());
        Value q = quadLoop.getInductionVar();
        Value acc = quadLoop.getRegionIterArgs()[0];

        // Compute element matrix entry
        Value phi_i = builder->create<memref::LoadOp>(
            loc, basis, ValueRange{i, q});
        Value phi_j = builder->create<memref::LoadOp>(
            loc, basis, ValueRange{j, q});
        Value weight = builder->create<memref::LoadOp>(
            loc, weights, ValueRange{q});

        // Integrand: phi_i * phi_j * weight
        Value prod = builder->create<arith::MulFOp>(loc, phi_i, phi_j);
        Value weighted = builder->create<arith::MulFOp>(loc, prod, weight);
        Value newAcc = builder->create<arith::AddFOp>(loc, acc, weighted);

        builder->create<scf::YieldOp>(loc, ValueRange{newAcc});
        builder->setInsertionPointAfter(quadLoop);

        // Store to element matrix
        Value elemValue = quadLoop.getResults()[0];
        builder->create<memref::StoreOp>(
            loc, elemValue, elementMatrices, ValueRange{elemIdx, i, j});

        // Get global indices
        Value globalI = builder->create<memref::LoadOp>(
            loc, connectivity, ValueRange{elemIdx, i});
        Value globalJ = builder->create<memref::LoadOp>(
            loc, connectivity, ValueRange{elemIdx, j});

        // Add to global matrix (should be atomic for thread safety)
        Value oldGlobal = builder->create<memref::LoadOp>(
            loc, globalMatrix, ValueRange{globalI, globalJ});
        Value newGlobal = builder->create<arith::AddFOp>(loc, oldGlobal, elemValue);
        builder->create<memref::StoreOp>(
            loc, newGlobal, globalMatrix, ValueRange{globalI, globalJ});

        // Close loops
        builder->setInsertionPointAfter(trialLoop);
        builder->setInsertionPointAfter(testLoop);
        builder->setInsertionPointAfter(elemLoop);
    }

public:
    //===------------------------------------------------------------------===//
    // Optimization Pipeline
    //===------------------------------------------------------------------===//

    void optimizeModule(int optimizationLevel = 2) {
        PassManager pm(context.get());

        // Level 0: No optimization
        if (optimizationLevel == 0) return;

        // Level 1: Basic optimizations
        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (optimizationLevel >= 2) {
            // Level 2: Standard optimizations
            pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
            pm.addNestedPass<func::FuncOp>(affine::createAffineScalarReplacementPass());
            pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());

            // Vectorization for SIMD
            pm.addNestedPass<func::FuncOp>(affine::createLoopUnrollPass());
        }

        if (optimizationLevel >= 3) {
            // Level 3: Aggressive optimizations
            pm.addNestedPass<func::FuncOp>(affine::createLoopTilingPass());
            pm.addNestedPass<func::FuncOp>(affine::createAffineDataCopyGenerationPass());

            // Sparsification
            SparsificationOptions sparseOpts;
            sparseOpts.enableRuntimeLibrary = true;
            pm.addPass(createSparsificationPass(sparseOpts));
        }

        // Run the pipeline
        if (failed(pm.run(module))) {
            llvm::errs() << "Optimization pipeline failed!\n";
        }
    }

    //===------------------------------------------------------------------===//
    // Code Generation
    //===------------------------------------------------------------------===//

    void lowerToLLVM() {
        PassManager pm(context.get());

        // Lower high-level constructs
        pm.addNestedPass<func::FuncOp>(createLowerAffinePass());
        pm.addPass(createConvertVectorToSCFPass());

        // Convert SCF to ControlFlow first
        pm.addPass(createSCFToControlFlowPass());

        // Convert to LLVM dialect
        ConvertVectorToLLVMPassOptions vecOpts;
        vecOpts.reassociateFPReductions = true;
        #ifdef __ARM_NEON
        vecOpts.armNeon = true;  // Enable NEON for Apple Silicon
        #endif

        pm.addPass(createConvertVectorToLLVMPass(vecOpts));
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());

        if (failed(pm.run(module))) {
            llvm::errs() << "Lowering to LLVM failed!\n";
        }
    }

    //===------------------------------------------------------------------===//
    // JIT Compilation and Execution
    //===------------------------------------------------------------------===//

    bool createJIT() {
        // First lower to LLVM if not already done
        lowerToLLVM();

        // Ensure all translations are registered
        DialectRegistry registry;
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerAllToLLVMIRTranslations(registry);
        context->appendDialectRegistry(registry);

        // Apply to module as well
        module.getContext()->appendDialectRegistry(registry);

        // Create execution engine with simpler settings to avoid analysis pass issues
        ExecutionEngineOptions options;

        // Don't use the optimizing transformer - it causes analysis pass registration issues
        // Just set the JIT optimization level
        options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

        // Create execution engine
        auto engineOrError = ExecutionEngine::create(module, options);
        if (!engineOrError) {
            llvm::errs() << "Failed to create ExecutionEngine\n";
            return false;
        }

        executionEngine = std::move(*engineOrError);
        return true;
    }

    // Get function pointer from JIT
    void* getCompiledFunction(StringRef funcName) {
        if (!executionEngine) {
            if (!createJIT())
                return nullptr;
        }

        auto result = executionEngine->lookupPacked(funcName);
        if (!result) {
            llvm::errs() << "Failed to lookup function: " << funcName << "\n";
            return nullptr;
        }

        // Extract raw function pointer from Expected
        return reinterpret_cast<void*>(*result);
    }

    // Execute JIT-compiled function
    void executeFunction(StringRef funcName, void** args, size_t numArgs) {
        if (!executionEngine) {
            if (!createJIT())
                return;
        }

        // Create MutableArrayRef from args
        if (args && numArgs > 0) {
            llvm::MutableArrayRef<void*> argArray(args, numArgs);
            auto result = executionEngine->invokePacked(funcName, argArray);
            if (result) {
                llvm::errs() << "Execution failed: " << result << "\n";
            }
        } else {
            auto result = executionEngine->invokePacked(funcName);
            if (result) {
                llvm::errs() << "Execution failed: " << result << "\n";
            }
        }
    }

    //===------------------------------------------------------------------===//
    // Utilities
    //===------------------------------------------------------------------===//

    bool verify() {
        return succeeded(mlir::verify(module));
    }

    void dump() {
        module.dump();
    }

    ModuleOp getModule() {
        return module;
    }
};

//===----------------------------------------------------------------------===//
// C API for Python Bindings
//===----------------------------------------------------------------------===//

extern "C" {
    void* mlir_compiler_create() {
        return new MLIRCompiler();
    }

    void mlir_compiler_destroy(void* compiler) {
        delete static_cast<MLIRCompiler*>(compiler);
    }

    void* mlir_compiler_generate_fem_assembly(
        void* compiler, int numElements, int dofsPerElement, int quadPoints) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        auto func = comp->generateFEMAssembly(numElements, dofsPerElement, quadPoints);
        return const_cast<void*>(func.getAsOpaquePointer());
    }

    void mlir_compiler_optimize(void* compiler, int level) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        comp->optimizeModule(level);
    }

    void mlir_compiler_lower_to_llvm(void* compiler) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        comp->lowerToLLVM();
    }

    bool mlir_compiler_verify(void* compiler) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        return comp->verify();
    }

    void mlir_compiler_dump(void* compiler) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        comp->dump();
    }

    void mlir_compiler_dump_after_lowering(void* compiler) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        comp->lowerToLLVM();
        comp->dump();
    }

    bool mlir_compiler_create_jit(void* compiler) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        return comp->createJIT();
    }

    void* mlir_compiler_get_function(void* compiler, const char* funcName) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        return comp->getCompiledFunction(funcName);
    }

    void mlir_compiler_execute(void* compiler, const char* funcName, void** args, int numArgs) {
        auto* comp = static_cast<MLIRCompiler*>(compiler);
        comp->executeFunction(funcName, args, static_cast<size_t>(numArgs));
    }
}

} // namespace mlir_backend
} // namespace firedrake
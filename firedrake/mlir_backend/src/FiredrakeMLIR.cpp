/*
 * Firedrake MLIR Native Python Extension
 * 
 * This provides direct C++ API access to MLIR, eliminating subprocess overhead
 * and enabling custom optimization passes.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// Dialect includes
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// Transform includes
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// LLVM includes
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

namespace py = pybind11;

namespace mlir {
namespace firedrake {

//===----------------------------------------------------------------------===//
// Context Manager
//===----------------------------------------------------------------------===//

class MLIRContextManager {
public:
    MLIRContextManager() {
        context = std::make_unique<MLIRContext>();
        
        // Register standard dialects
        context->loadDialect<affine::AffineDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<func::FuncDialect>();
        context->loadDialect<linalg::LinalgDialect>();
        context->loadDialect<memref::MemRefDialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<tensor::TensorDialect>();
        
        // Register Firedrake-specific dialects when they are defined
        // For now, using standard MLIR dialects is sufficient for FEM operations
        // context->loadDialect<fem::FEMDialect>();
        // context->loadDialect<gem::GEMDialect>();
    }
    
    MLIRContext* getContext() { return context.get(); }
    
private:
    std::unique_ptr<MLIRContext> context;
};

//===----------------------------------------------------------------------===//
// Module Builder
//===----------------------------------------------------------------------===//

class ModuleBuilder {
public:
    ModuleBuilder(MLIRContextManager& ctx) 
        : context(ctx.getContext()), 
          module(ModuleOp::create(UnknownLoc::get(context))),
          builder(context) {
        builder.setInsertionPointToEnd(module.getBody());
    }
    
    // Create a function
    func::FuncOp createFunction(const std::string& name,
                                const std::vector<int64_t>& argShapes,
                                const std::string& elementType = "f64") {
        // Build function type
        SmallVector<Type, 4> argTypes;
        Type elemType = elementType == "f64" ? 
            builder.getF64Type() : builder.getF32Type();
        
        for (const auto& shape : argShapes) {
            if (shape == -1) {
                // Scalar argument
                argTypes.push_back(elemType);
            } else {
                // Memref argument
                argTypes.push_back(MemRefType::get({shape, shape}, elemType));
            }
        }
        
        auto funcType = builder.getFunctionType(argTypes, {});
        auto func = func::FuncOp::create(builder.getUnknownLoc(), name, funcType);
        module.push_back(func);
        
        // Create entry block
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        return func;
    }
    
    // Create constants
    Value createConstant(double value) {
        return builder.create<arith::ConstantOp>(
            builder.getUnknownLoc(),
            builder.getF64FloatAttr(value)
        );
    }
    
    Value createIndex(int64_t value) {
        return builder.create<arith::ConstantIndexOp>(
            builder.getUnknownLoc(), value
        );
    }
    
    // Create loops
    scf::ForOp createForLoop(Value start, Value end, Value step) {
        return builder.create<scf::ForOp>(
            builder.getUnknownLoc(), start, end, step
        );
    }
    
    // Create memory operations
    void createStore(Value value, Value memref, ValueRange indices) {
        builder.create<memref::StoreOp>(
            builder.getUnknownLoc(), value, memref, indices
        );
    }
    
    Value createLoad(Value memref, ValueRange indices) {
        return builder.create<memref::LoadOp>(
            builder.getUnknownLoc(), memref, indices
        );
    }
    
    // Arithmetic operations
    Value createAdd(Value lhs, Value rhs) {
        return builder.create<arith::AddFOp>(
            builder.getUnknownLoc(), lhs, rhs
        );
    }
    
    Value createMul(Value lhs, Value rhs) {
        return builder.create<arith::MulFOp>(
            builder.getUnknownLoc(), lhs, rhs
        );
    }
    
    // Get module as string
    std::string getModuleStr() {
        std::string str;
        llvm::raw_string_ostream os(str);
        module.print(os);
        return str;
    }
    
    ModuleOp getModule() { return module; }
    OpBuilder& getBuilder() { return builder; }
    
private:
    MLIRContext* context;
    ModuleOp module;
    OpBuilder builder;
};

//===----------------------------------------------------------------------===//
// Pass Manager
//===----------------------------------------------------------------------===//

class OptimizationPassManager {
public:
    OptimizationPassManager(MLIRContext* context) : pm(context) {}
    
    void addStandardOptimizations() {
        // Standard optimization passes
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(affine::createAffineScalarReplacementPass());
        pm.addPass(createLoopInvariantCodeMotionPass());
    }
    
    void addAggressiveOptimizations() {
        addStandardOptimizations();
        pm.addPass(affine::createLoopFusionPass());
        pm.addPass(affine::createLoopTilingPass());
        pm.addPass(affine::createAffineVectorize());
    }
    
    void addLoweringPasses() {
        pm.addPass(createLowerAffinePass());
        pm.addPass(createSCFToControlFlowPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());
    }
    
    bool run(ModuleOp module) {
        return succeeded(pm.run(module));
    }
    
private:
    PassManager pm;
};

//===----------------------------------------------------------------------===//
// Custom Firedrake Passes
//===----------------------------------------------------------------------===//

// Sum Factorization Pass
struct SumFactorizationPass : public PassWrapper<SumFactorizationPass, 
                                                  OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SumFactorizationPass)
    
    void runOnOperation() override {
        func::FuncOp func = getOperation();
        
        // Walk through all nested loops
        func.walk([&](scf::ForOp forOp) {
            // Look for sum factorization opportunities
            // This is a simplified version - real implementation would be more complex
            
            // Pattern: sum_i sum_j A[i] * B[j] => (sum_i A[i]) * (sum_j B[j])
            if (auto nestedFor = dyn_cast<scf::ForOp>(forOp.getBody()->front())) {
                // Check if we can factor the sums
                // ... analysis code ...
            }
        });
    }
    
    StringRef getArgument() const final { return "firedrake-sum-factorization"; }
    StringRef getDescription() const final { 
        return "Apply sum factorization optimization"; 
    }
};

// Delta Elimination Pass
struct DeltaEliminationPass : public PassWrapper<DeltaEliminationPass,
                                                  OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeltaEliminationPass)
    
    void runOnOperation() override {
        func::FuncOp func = getOperation();
        
        // Walk through operations looking for delta patterns
        func.walk([&](Operation* op) {
            // Look for patterns like: delta(i,j) * expr(j) => expr(i)
            // This would require custom delta operations in the dialect
        });
    }
    
    StringRef getArgument() const final { return "firedrake-delta-elimination"; }
    StringRef getDescription() const final { 
        return "Eliminate Kronecker delta operations"; 
    }
};

//===----------------------------------------------------------------------===//
// Python Bindings
//===----------------------------------------------------------------------===//

// Helper to convert numpy dtype to MLIR element type string
std::string numpyDtypeToMLIR(py::dtype dt) {
    if (dt.is(py::dtype::of<double>())) return "f64";
    if (dt.is(py::dtype::of<float>())) return "f32";
    if (dt.is(py::dtype::of<int64_t>())) return "i64";
    if (dt.is(py::dtype::of<int32_t>())) return "i32";
    return "f64"; // default
}

PYBIND11_MODULE(firedrake_mlir_native, m) {
    m.doc() = "Firedrake MLIR Native Extension - Direct C++ API access";
    
    // Context Manager
    py::class_<MLIRContextManager>(m, "Context")
        .def(py::init<>())
        .def("__repr__", [](const MLIRContextManager&) {
            return "<MLIRContext with standard dialects>";
        });
    
    // Module Builder
    py::class_<ModuleBuilder>(m, "ModuleBuilder")
        .def(py::init<MLIRContextManager&>())
        .def("create_function", &ModuleBuilder::createFunction,
             py::arg("name"),
             py::arg("arg_shapes"),
             py::arg("element_type") = "f64",
             "Create a function with given argument shapes")
        .def("constant", py::overload_cast<double>(&ModuleBuilder::createConstant),
             "Create a floating point constant")
        .def("index", &ModuleBuilder::createIndex,
             "Create an index constant")
        .def("for_loop", [](ModuleBuilder& self, int64_t start, int64_t end, int64_t step) {
             auto startIdx = self.createIndex(start);
             auto endIdx = self.createIndex(end);
             auto stepIdx = self.createIndex(step);
             return self.createForLoop(startIdx, endIdx, stepIdx);
         }, "Create a for loop")
        .def("add", &ModuleBuilder::createAdd, "Add two values")
        .def("mul", &ModuleBuilder::createMul, "Multiply two values")
        .def("get_mlir", &ModuleBuilder::getModuleStr,
             "Get MLIR module as string")
        .def("__str__", &ModuleBuilder::getModuleStr);
    
    // Pass Manager
    py::class_<OptimizationPassManager>(m, "PassManager")
        .def(py::init<MLIRContext*>())
        .def("add_standard_optimizations", 
             &OptimizationPassManager::addStandardOptimizations)
        .def("add_aggressive_optimizations",
             &OptimizationPassManager::addAggressiveOptimizations)
        .def("add_lowering_passes",
             &OptimizationPassManager::addLoweringPasses);
    
    // Register custom passes
    m.def("register_firedrake_passes", []() {
        PassRegistration<SumFactorizationPass>();
        PassRegistration<DeltaEliminationPass>();
    });
    
    // High-level kernel generation
    m.def("generate_mass_matrix_kernel", 
        [](MLIRContextManager& ctx, int n_dofs) -> std::string {
            ModuleBuilder builder(ctx);
            
            // Create kernel function
            auto func = builder.createFunction(
                "mass_matrix_kernel",
                {n_dofs}, // Single n_dofs x n_dofs matrix
                "f64"
            );
            
            // Generate nested loops
            auto one = builder.createConstant(1.0);
            auto qweight = builder.createConstant(0.25);
            
            auto c0 = builder.createIndex(0);
            auto cn = builder.createIndex(n_dofs);
            auto c1 = builder.createIndex(1);
            
            // Get matrix argument
            auto args = func.getArguments();
            auto matrix = args[0];
            
            // Outer loop over i
            auto outerLoop = builder.createForLoop(c0, cn, c1);
            builder.getBuilder().setInsertionPointToStart(outerLoop.getBody());
            auto i = outerLoop.getInductionVar();
            
            // Inner loop over j
            auto innerLoop = builder.createForLoop(c0, cn, c1);
            builder.getBuilder().setInsertionPointToStart(innerLoop.getBody());
            auto j = innerLoop.getInductionVar();
            
            // Simplified computation: M[i,j] = 1.0 * 0.25
            auto result = builder.createMul(one, qweight);
            builder.createStore(result, matrix, {i, j});
            
            // Add return
            builder.getBuilder().setInsertionPointAfter(outerLoop);
            builder.getBuilder().create<func::ReturnOp>(
                builder.getBuilder().getUnknownLoc()
            );
            
            return builder.getModuleStr();
        },
        py::arg("context"),
        py::arg("n_dofs") = 3,
        "Generate a mass matrix assembly kernel"
    );
    
    // Direct optimization function
    m.def("optimize_mlir", 
        [](const std::string& mlir_str, const std::string& mode) -> std::string {
            MLIRContextManager ctx;
            auto context = ctx.getContext();
            
            // Parse MLIR
            auto module = parseSourceString<ModuleOp>(mlir_str, context);
            if (!module) {
                throw std::runtime_error("Failed to parse MLIR");
            }
            
            // Create pass manager
            OptimizationPassManager pm(context);
            
            if (mode == "standard") {
                pm.addStandardOptimizations();
            } else if (mode == "aggressive") {
                pm.addAggressiveOptimizations();
            } else if (mode == "lower") {
                pm.addLoweringPasses();
            }
            
            // Run passes
            if (!pm.run(*module)) {
                throw std::runtime_error("Optimization failed");
            }
            
            // Return optimized MLIR
            std::string result;
            llvm::raw_string_ostream os(result);
            module->print(os);
            return result;
        },
        py::arg("mlir_str"),
        py::arg("mode") = "standard",
        "Optimize MLIR code directly without subprocess"
    );
    
    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("LLVM_VERSION") = LLVM_VERSION_STRING;
}

} // namespace firedrake
} // namespace mlir
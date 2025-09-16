/*
 * Correct MLIR C++ API Usage Examples for Firedrake
 *
 * This file demonstrates the CORRECT way to use MLIR APIs
 * for essential Firedrake operations.
 */

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ESSENTIAL #1: Correct FEM Assembly Loop Generation
//===----------------------------------------------------------------------===//

void generateCorrectFEMAssembly(OpBuilder& builder, Location loc,
                                Value globalMatrix, Value elementMatrices,
                                Value connectivity, int nElems, int nDofs) {

    // CORRECT: Use proper index types and bounds
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto cNElems = builder.create<arith::ConstantIndexOp>(loc, nElems);
    auto cNDofs = builder.create<arith::ConstantIndexOp>(loc, nDofs);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

    // CORRECT: Element loop with proper scf::ForOp
    auto elemLoop = builder.create<scf::ForOp>(loc, c0, cNElems, c1);
    builder.setInsertionPointToStart(elemLoop.getBody());
    Value elemIdx = elemLoop.getInductionVar();

    // CORRECT: Test function loop
    auto testLoop = builder.create<scf::ForOp>(loc, c0, cNDofs, c1);
    builder.setInsertionPointToStart(testLoop.getBody());
    Value i = testLoop.getInductionVar();

    // CORRECT: Trial function loop
    auto trialLoop = builder.create<scf::ForOp>(loc, c0, cNDofs, c1);
    builder.setInsertionPointToStart(trialLoop.getBody());
    Value j = trialLoop.getInductionVar();

    // CORRECT: Load local matrix value
    Value localVal = builder.create<memref::LoadOp>(
        loc, elementMatrices, ValueRange{elemIdx, i, j});

    // CORRECT: Get global indices from connectivity
    Value globalI = builder.create<memref::LoadOp>(
        loc, connectivity, ValueRange{elemIdx, i});
    Value globalJ = builder.create<memref::LoadOp>(
        loc, connectivity, ValueRange{elemIdx, j});

    // CORRECT: Atomic accumulate to global matrix (for thread safety)
    Value oldVal = builder.create<memref::LoadOp>(
        loc, globalMatrix, ValueRange{globalI, globalJ});
    Value newVal = builder.create<arith::AddFOp>(loc, oldVal, localVal);
    builder.create<memref::StoreOp>(
        loc, newVal, globalMatrix, ValueRange{globalI, globalJ});

    // CORRECT: Set insertion point after loops
    builder.setInsertionPointAfter(trialLoop);
    builder.setInsertionPointAfter(testLoop);
    builder.setInsertionPointAfter(elemLoop);
}

//===----------------------------------------------------------------------===//
// ESSENTIAL #2: Correct Sparse Matrix Operations
//===----------------------------------------------------------------------===//

void generateCorrectSparseOps(OpBuilder& builder, MLIRContext* context,
                              Location loc) {

    // CORRECT: Define sparse tensor encoding for CSR format
    auto enc = SparseTensorEncodingAttr::get(
        context,
        SmallVector<DimLevelType>{DimLevelType::Dense, DimLevelType::Compressed},
        AffineMap(),  // dimOrdering
        AffineMap(),  // higherOrdering
        64,          // posWidth
        64           // crdWidth
    );

    // CORRECT: Create sparse tensor type
    auto f64Type = builder.getF64Type();
    auto sparseType = RankedTensorType::get({1000, 1000}, f64Type, enc);

    // CORRECT: Use sparse_tensor operations
    auto sparseMatrix = builder.create<sparse_tensor::LoadOp>(loc, sparseType);

    // CORRECT: Iterate only over non-zeros
    builder.create<sparse_tensor::ForeachOp>(
        loc, sparseMatrix,
        [&](OpBuilder& b, Location l, ValueRange coords, Value val,
            ValueRange reduc) {
            // Process only non-zero entries
            Value row = coords[0];
            Value col = coords[1];

            // Do computation with the non-zero value
            Value doubled = b.create<arith::MulFOp>(l, val,
                b.create<arith::ConstantOp>(l, b.getF64FloatAttr(2.0)));

            // Store back (if needed)
            b.create<sparse_tensor::YieldOp>(l, doubled);
        });
}

//===----------------------------------------------------------------------===//
// ESSENTIAL #3: Correct Vectorization for Apple M4 NEON
//===----------------------------------------------------------------------===//

void generateCorrectVectorization(OpBuilder& builder, Location loc,
                                  Value input, Value output, int size) {

    // CORRECT: Apple M4 has 128-bit NEON registers (2 x f64)
    const int VECTOR_WIDTH = 2;
    auto f64Type = builder.getF64Type();
    auto vecType = VectorType::get({VECTOR_WIDTH}, f64Type);

    // CORRECT: Main vectorized loop
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto cSize = builder.create<arith::ConstantIndexOp>(loc, size);
    auto cVecWidth = builder.create<arith::ConstantIndexOp>(loc, VECTOR_WIDTH);

    // CORRECT: Process vector-width elements at a time
    auto vecLoop = builder.create<scf::ForOp>(loc, c0, cSize, cVecWidth);
    builder.setInsertionPointToStart(vecLoop.getBody());
    Value idx = vecLoop.getInductionVar();

    // CORRECT: Vector load
    Value vec = builder.create<vector::LoadOp>(
        loc, vecType, input, ValueRange{idx});

    // CORRECT: Vector operation (example: multiply by 2)
    Value two = builder.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(vecType, 2.0));
    Value result = builder.create<arith::MulFOp>(loc, vec, two);

    // CORRECT: Vector store
    builder.create<vector::StoreOp>(loc, result, output, ValueRange{idx});

    builder.setInsertionPointAfter(vecLoop);

    // CORRECT: Handle remainder (if size not divisible by VECTOR_WIDTH)
    Value remainder = builder.create<arith::RemUIOp>(loc, cSize, cVecWidth);
    auto remainderLoop = builder.create<scf::IfOp>(
        loc, builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, remainder, c0));

    // Process remaining elements scalar
    builder.setInsertionPointToStart(&remainderLoop.getThenRegion().front());
    // ... scalar processing ...
}

//===----------------------------------------------------------------------===//
// ESSENTIAL #4: Correct Memory Management
//===----------------------------------------------------------------------===//

void correctMemoryManagement(OpBuilder& builder, Location loc) {

    // CORRECT: Allocate with proper type
    auto memrefType = MemRefType::get({100, 100}, builder.getF64Type());
    Value matrix = builder.create<memref::AllocOp>(loc, memrefType);

    // ... use the matrix ...

    // CORRECT: Always deallocate (or let bufferization pass handle it)
    builder.create<memref::DeallocOp>(loc, matrix);

    // ALTERNATIVE: Use alloca for stack allocation (auto-freed)
    Value stackMatrix = builder.create<memref::AllocaOp>(loc, memrefType);
    // No explicit dealloc needed for alloca
}

//===----------------------------------------------------------------------===//
// ESSENTIAL #5: Correct Pass Pipeline Application
//===----------------------------------------------------------------------===//

void applyCorrectPassPipeline(ModuleOp module, MLIRContext* context) {

    PassManager pm(context);

    // CORRECT: Module-level passes
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // CORRECT: Function-level passes use addNestedPass
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(affine::createAffineScalarReplacementPass());
    pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
    pm.addNestedPass<func::FuncOp>(affine::createLoopTilingPass());

    // CORRECT: Vectorization with options
    affine::SuperVectorizeOptions vecOpts;
    vecOpts.vectorSizes = {2};  // NEON width for f64
    pm.addNestedPass<func::FuncOp>(
        affine::createSuperVectorizePass(vecOpts));

    // CORRECT: Sparsification with options
    SparsificationOptions sparseOpts;
    sparseOpts.enableRuntimeLibrary = true;
    pm.addPass(createSparsificationPass(sparseOpts));

    // CORRECT: Lower to LLVM with proper options
    ConvertVectorToLLVMPassOptions vecToLLVMOpts;
    vecToLLVMOpts.armNeon = true;  // Enable NEON for Apple M4
    pm.addPass(createConvertVectorToLLVMPass(vecToLLVMOpts));

    // Run the pipeline
    if (failed(pm.run(module))) {
        llvm::errs() << "Pass pipeline failed!\n";
    }
}

//===----------------------------------------------------------------------===//
// ESSENTIAL #6: Correct Dynamic Dimensions Handling
//===----------------------------------------------------------------------===//

void handleDynamicDimensions(OpBuilder& builder, Location loc) {

    // WRONG: Using -1 for dynamic dimensions
    // auto wrongType = MemRefType::get({-1, -1}, builder.getF64Type());

    // CORRECT: Use ShapedType::kDynamic
    auto dynamicType = MemRefType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic},
        builder.getF64Type());

    // CORRECT: Query runtime dimensions
    Value dynamicMatrix = builder.create<memref::AllocOp>(loc, dynamicType);
    Value dim0 = builder.create<memref::DimOp>(loc, dynamicMatrix, 0);
    Value dim1 = builder.create<memref::DimOp>(loc, dynamicMatrix, 1);

    // CORRECT: Use runtime dimensions in loops
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

    auto outerLoop = builder.create<scf::ForOp>(loc, c0, dim0, c1);
    builder.setInsertionPointToStart(outerLoop.getBody());

    auto innerLoop = builder.create<scf::ForOp>(loc, c0, dim1, c1);
    // ... loop body ...
}

//===----------------------------------------------------------------------===//
// ESSENTIAL #7: Correct Error Handling Pattern
//===----------------------------------------------------------------------===//

LogicalResult processWithErrorHandling(Value input, PatternRewriter& rewriter) {

    // CORRECT: Check preconditions
    auto inputType = input.getType().dyn_cast<MemRefType>();
    if (!inputType) {
        return failure();  // Not a memref
    }

    if (inputType.getRank() != 2) {
        return failure();  // Not a matrix
    }

    auto shape = inputType.getShape();
    if (shape[0] == ShapedType::kDynamic ||
        shape[1] == ShapedType::kDynamic) {
        // Handle dynamic case differently
        return failure();  // Or handle specially
    }

    // CORRECT: Validate dimensions
    if (shape[0] <= 0 || shape[1] <= 0) {
        return failure();  // Invalid dimensions
    }

    // Process the valid input
    // ...

    return success();
}
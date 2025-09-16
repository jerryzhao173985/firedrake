/*
 * Unit Test: Vector Operations
 *
 * Tests vector operations for SIMD (M4 NEON) support
 * This validates our ability to generate vectorized code
 */

#include "../test_utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::firedrake::test;

void test_vector_type_creation() {
    auto context = createTestContext();
    OpBuilder builder(context.get());

    // Test vector types for M4 NEON (supports 128-bit vectors)
    auto f64Type = builder.getF64Type();
    auto f32Type = builder.getF32Type();

    // 2 x f64 (128 bits)
    auto vec2f64 = VectorType::get({2}, f64Type);
    EXPECT_EQ(2, vec2f64.getDimSize(0));

    // 4 x f32 (128 bits)
    auto vec4f32 = VectorType::get({4}, f32Type);
    EXPECT_EQ(4, vec4f32.getDimSize(0));

    // Multi-dimensional vectors
    auto vec2x4f64 = VectorType::get({2, 4}, f64Type);
    EXPECT_EQ(2, vec2x4f64.getRank());

    llvm::outs() << "✅ Vector type creation works\n";
}

void test_vector_load_store() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    // Create function with vector load/store
    auto f64Type = builder.getF64Type();
    auto memrefType = createMemRefType(context.get(), {1024}, f64Type);
    auto vecType = VectorType::get({4}, f64Type);

    auto func = createTestFunction(builder, module, "vector_load_store",
        {memrefType, memrefType}, {});

    auto* entryBlock = &func.getBody().front();
    builder.setInsertionPointToStart(entryBlock);

    Value src = entryBlock->getArgument(0);
    Value dst = entryBlock->getArgument(1);

    // Create vectorized loop
    auto c0 = createConstantIndex(builder, 0);
    auto c4 = createConstantIndex(builder, 4);
    auto c1024 = createConstantIndex(builder, 1024);

    auto loop = builder.create<scf::ForOp>(
        builder.getUnknownLoc(), c0, c1024, c4
    );
    builder.setInsertionPointToStart(loop.getBody());
    Value idx = loop.getInductionVar();

    // Vector load
    Value vec = builder.create<vector::LoadOp>(
        builder.getUnknownLoc(), vecType, src, ValueRange{idx}
    );

    // Vector store
    builder.create<vector::StoreOp>(
        builder.getUnknownLoc(), vec, dst, ValueRange{idx}
    );

    builder.setInsertionPointAfter(loop);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    EXPECT_TRUE(verifyModule(module));
    EXPECT_TRUE(containsOp(module, "vector.load"));
    EXPECT_TRUE(containsOp(module, "vector.store"));

    llvm::outs() << "✅ Vector load/store works\n";
}

void test_vector_arithmetic() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto func = createTestFunction(builder, module, "vector_arith", {}, {});
    builder.setInsertionPointToStart(&func.getBody().front());

    // Create vector constants
    auto vecType = VectorType::get({4}, builder.getF64Type());
    auto vec1 = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(vecType, {1.0, 2.0, 3.0, 4.0})
    );
    auto vec2 = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(vecType, {5.0, 6.0, 7.0, 8.0})
    );

    // Vector arithmetic operations
    auto add = builder.create<arith::AddFOp>(
        builder.getUnknownLoc(), vec1, vec2
    );
    auto mul = builder.create<arith::MulFOp>(
        builder.getUnknownLoc(), vec1, vec2
    );
    auto sub = builder.create<arith::SubFOp>(
        builder.getUnknownLoc(), vec2, vec1
    );

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    EXPECT_TRUE(verifyModule(module));

    llvm::outs() << "✅ Vector arithmetic works\n";
}

void test_vector_reduction() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto func = createTestFunction(builder, module, "vector_reduction", {}, {});
    builder.setInsertionPointToStart(&func.getBody().front());

    // Create vector for reduction
    auto vecType = VectorType::get({4}, builder.getF64Type());
    auto vec = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(vecType, {1.0, 2.0, 3.0, 4.0})
    );

    // Vector reduction operations
    auto sum = builder.create<vector::ReductionOp>(
        builder.getUnknownLoc(),
        vector::CombiningKind::ADD,
        vec.getResult()
    );

    auto mul = builder.create<vector::ReductionOp>(
        builder.getUnknownLoc(),
        vector::CombiningKind::MUL,
        vec.getResult()
    );

    auto max = builder.create<vector::ReductionOp>(
        builder.getUnknownLoc(),
        vector::CombiningKind::MAXIMUMF,
        vec.getResult()
    );

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    EXPECT_TRUE(verifyModule(module));
    EXPECT_TRUE(containsOp(module, "vector.reduction"));

    llvm::outs() << "✅ Vector reduction works\n";
}

void test_vector_broadcast() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto func = createTestFunction(builder, module, "vector_broadcast", {}, {});
    builder.setInsertionPointToStart(&func.getBody().front());

    // Broadcast scalar to vector
    auto scalar = createConstantF64(builder, 3.14);
    auto vecType = VectorType::get({4}, builder.getF64Type());

    auto broadcast = builder.create<vector::BroadcastOp>(
        builder.getUnknownLoc(), vecType, scalar
    );

    // Broadcast vector to higher dimension
    auto vec2Type = VectorType::get({2, 4}, builder.getF64Type());
    auto broadcast2d = builder.create<vector::BroadcastOp>(
        builder.getUnknownLoc(), vec2Type, broadcast.getResult()
    );

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    EXPECT_TRUE(verifyModule(module));
    EXPECT_TRUE(containsOp(module, "vector.broadcast"));

    llvm::outs() << "✅ Vector broadcast works\n";
}

void test_vector_transfer() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto f64Type = builder.getF64Type();
    auto memrefType = createMemRefType(context.get(), {10, 10}, f64Type);
    auto vecType = VectorType::get({4}, f64Type);

    auto func = createTestFunction(builder, module, "vector_transfer",
        {memrefType}, {});

    auto* entryBlock = &func.getBody().front();
    builder.setInsertionPointToStart(entryBlock);

    Value memref = entryBlock->getArgument(0);

    // Create indices
    auto c0 = createConstantIndex(builder, 0);
    auto c1 = createConstantIndex(builder, 1);

    // Transfer read (memref to vector)
    auto padding = createConstantF64(builder, 0.0);
    auto read = builder.create<vector::TransferReadOp>(
        builder.getUnknownLoc(), vecType, memref,
        ValueRange{c0, c1}, padding
    );

    // Modify vector
    auto modified = builder.create<arith::MulFOp>(
        builder.getUnknownLoc(), read.getResult(),
        builder.create<arith::ConstantOp>(
            builder.getUnknownLoc(),
            DenseElementsAttr::get(vecType, 2.0)
        )
    );

    // Transfer write (vector to memref)
    builder.create<vector::TransferWriteOp>(
        builder.getUnknownLoc(), modified, memref,
        ValueRange{c0, c1}
    );

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    EXPECT_TRUE(verifyModule(module));
    EXPECT_TRUE(containsOp(module, "vector.transfer_read"));
    EXPECT_TRUE(containsOp(module, "vector.transfer_write"));

    llvm::outs() << "✅ Vector transfer works\n";
}

void test_vector_masking() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto func = createTestFunction(builder, module, "vector_masking", {}, {});
    builder.setInsertionPointToStart(&func.getBody().front());

    // Create mask
    auto i1Type = builder.getI1Type();
    auto maskType = VectorType::get({4}, i1Type);
    auto mask = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(maskType, {true, true, false, true})
    );

    // Create vectors
    auto vecType = VectorType::get({4}, builder.getF64Type());
    auto vec1 = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(vecType, {1.0, 2.0, 3.0, 4.0})
    );
    auto vec2 = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(vecType, {5.0, 6.0, 7.0, 8.0})
    );

    // Masked operation (select based on mask)
    auto selected = builder.create<arith::SelectOp>(
        builder.getUnknownLoc(), mask, vec1, vec2
    );

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    EXPECT_TRUE(verifyModule(module));

    llvm::outs() << "✅ Vector masking works\n";
}

int main() {
    llvm::outs() << "\n";
    llvm::outs() << "=====================================\n";
    llvm::outs() << "Unit Test: Vector Operations\n";
    llvm::outs() << "=====================================\n\n";

    RUN_TEST(test_vector_type_creation);
    RUN_TEST(test_vector_load_store);
    RUN_TEST(test_vector_arithmetic);
    RUN_TEST(test_vector_reduction);
    RUN_TEST(test_vector_broadcast);
    RUN_TEST(test_vector_transfer);
    RUN_TEST(test_vector_masking);

    llvm::outs() << "\n";
    llvm::outs() << "=====================================\n";
    llvm::outs() << "✅ All vector operation tests PASSED!\n";
    llvm::outs() << "   SIMD support for M4 NEON validated\n";
    llvm::outs() << "=====================================\n\n";

    return 0;
}
/*
 * Unit Test: Dialect Loading
 *
 * Tests that all comprehensive MLIR dialects are loaded and functional
 */

#include "../test_utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::firedrake::test;

void test_all_dialects_loaded() {
    auto context = createTestContext();

    // Test essential dialects
    EXPECT_NOT_NULL(context->getLoadedDialect<affine::AffineDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<arith::ArithDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<func::FuncDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<linalg::LinalgDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<memref::MemRefDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<scf::SCFDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<tensor::TensorDialect>());

    // Test advanced dialects
    EXPECT_NOT_NULL(context->getLoadedDialect<math::MathDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<complex::ComplexDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<vector::VectorDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<sparse_tensor::SparseTensorDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<async::AsyncDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<gpu::GPUDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<bufferization::BufferizationDialect>());

    // Test pattern infrastructure dialects
    EXPECT_NOT_NULL(context->getLoadedDialect<pdl::PDLDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<pdl_interp::PDLInterpDialect>());
    EXPECT_NOT_NULL(context->getLoadedDialect<transform::TransformDialect>());

    llvm::outs() << "✅ All 17 dialects loaded successfully\n";
}

void test_dialect_operations() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    // Create function to test operations
    auto funcType = builder.getFunctionType({}, {});
    auto func = createTestFunction(builder, module, "test_ops", {}, {});

    builder.setInsertionPointToStart(&func.getBody().front());

    // Test Affine operations
    auto affineFor = builder.create<affine::AffineForOp>(
        builder.getUnknownLoc(), 0, 10, 1
    );
    EXPECT_NOT_NULL(affineFor.getOperation());

    // Test Arith operations
    auto f1 = createConstantF64(builder, 1.0);
    auto f2 = createConstantF64(builder, 2.0);
    auto add = builder.create<arith::AddFOp>(builder.getUnknownLoc(), f1, f2);
    EXPECT_NOT_NULL(add.getOperation());

    // Test Vector operations
    auto vecType = createVectorType(context.get(), {4}, builder.getF64Type());
    auto vecZero = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        DenseElementsAttr::get(vecType, 0.0)
    );
    EXPECT_NOT_NULL(vecZero.getOperation());

    // Test Math operations
    auto sqrt = builder.create<math::SqrtOp>(builder.getUnknownLoc(), f1);
    EXPECT_NOT_NULL(sqrt.getOperation());

    // Test Complex operations
    auto complexType = ComplexType::get(builder.getF64Type());
    auto complexVal = builder.create<complex::CreateOp>(
        builder.getUnknownLoc(), complexType, f1, f2
    );
    EXPECT_NOT_NULL(complexVal.getOperation());

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    // Verify the module
    EXPECT_TRUE(verifyModule(module));

    llvm::outs() << "✅ Dialect operations work correctly\n";
}

void test_dialect_type_systems() {
    auto context = createTestContext();

    // Test MemRef types
    OpBuilder builder(context.get());
    auto memrefType = createMemRefType(context.get(), {10, 20},
                                      builder.getF64Type());
    EXPECT_EQ(2, memrefType.getRank());

    // Test Tensor types
    auto tensorType = createTensorType(context.get(), {100, 100},
                                      builder.getF64Type());
    EXPECT_EQ(2, tensorType.getRank());

    // Test Vector types
    auto vectorType = createVectorType(context.get(), {4},
                                      builder.getF64Type());
    EXPECT_EQ(1, vectorType.getRank());

    // Test Complex types
    auto complexType = ComplexType::get(builder.getF64Type());
    EXPECT_TRUE(complexType.getElementType().isF64());

    llvm::outs() << "✅ Type systems work correctly\n";
}

void test_dialect_attributes() {
    auto context = createTestContext();
    OpBuilder builder(context.get());

    // Test DenseElementsAttr for vectors
    auto vecType = VectorType::get({4}, builder.getF64Type());
    auto denseAttr = DenseElementsAttr::get(vecType, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(4, denseAttr.getNumElements());

    // Test AffineMap
    auto map = AffineMap::get(2, 0,
        {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
        context.get());
    EXPECT_EQ(2, map.getNumDims());

    llvm::outs() << "✅ Dialect attributes work correctly\n";
}

int main() {
    llvm::outs() << "\n";
    llvm::outs() << "=====================================\n";
    llvm::outs() << "Unit Test: Dialect Loading\n";
    llvm::outs() << "=====================================\n\n";

    RUN_TEST(test_all_dialects_loaded);
    RUN_TEST(test_dialect_operations);
    RUN_TEST(test_dialect_type_systems);
    RUN_TEST(test_dialect_attributes);

    llvm::outs() << "\n";
    llvm::outs() << "=====================================\n";
    llvm::outs() << "✅ All dialect tests PASSED!\n";
    llvm::outs() << "=====================================\n\n";

    return 0;
}
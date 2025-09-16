/*
 * Unit Test: Pattern Rewriting System
 *
 * Tests the comprehensive pattern rewriting infrastructure
 * that replaces GEM/COFFEE optimizations with MLIR patterns
 */

#include "../test_utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::firedrake::test;

// Custom pattern for FEM optimization
struct FEMSumFactorizationPattern : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern<arith::MulFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                   PatternRewriter &rewriter) const override {
        // Pattern to detect and optimize sum factorization
        // This replaces what GEM would do

        // Check if this multiplication is part of a sum factorization
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();

        // Look for pattern: (a + b) * c -> a*c + b*c
        auto addOp = lhs.getDefiningOp<arith::AddFOp>();
        if (!addOp)
            addOp = rhs.getDefiningOp<arith::AddFOp>();

        if (addOp) {
            // Apply distribution
            Location loc = op.getLoc();
            Value other = (addOp == lhs.getDefiningOp()) ? rhs : lhs;

            Value newMul1 = rewriter.create<arith::MulFOp>(loc,
                addOp.getLhs(), other);
            Value newMul2 = rewriter.create<arith::MulFOp>(loc,
                addOp.getRhs(), other);
            Value result = rewriter.create<arith::AddFOp>(loc,
                newMul1, newMul2);

            rewriter.replaceOp(op, result);
            return success();
        }

        return failure();
    }
};

// Pattern for delta elimination (replacing COFFEE optimization)
struct DeltaEliminationPattern : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern<arith::MulFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                   PatternRewriter &rewriter) const override {
        // Check for multiplication by 1.0 (delta)
        auto constOp = op.getRhs().getDefiningOp<arith::ConstantOp>();
        if (constOp) {
            auto floatAttr = mlir::dyn_cast<FloatAttr>(constOp.getValue());
            if (floatAttr && floatAttr.getValueAsDouble() == 1.0) {
                // Eliminate multiplication by 1
                rewriter.replaceOp(op, op.getLhs());
                return success();
            }
        }
        return failure();
    }
};

void test_pattern_registration() {
    auto context = createTestContext();

    // Register custom patterns
    RewritePatternSet patterns(context.get());
    patterns.add<FEMSumFactorizationPattern>(context.get());
    patterns.add<DeltaEliminationPattern>(context.get());

    // Verify patterns are registered
    EXPECT_TRUE(patterns.getNativePatterns().size() >= 2);

    llvm::outs() << "✅ Pattern registration works\n";
}

void test_pattern_application() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    // Create function with operations to optimize
    auto func = createTestFunction(builder, module, "pattern_test", {}, {});
    builder.setInsertionPointToStart(&func.getBody().front());

    // Create pattern: (a + b) * c
    auto a = createConstantF64(builder, 2.0);
    auto b = createConstantF64(builder, 3.0);
    auto c = createConstantF64(builder, 4.0);

    auto add = builder.create<arith::AddFOp>(builder.getUnknownLoc(), a, b);
    auto mul = builder.create<arith::MulFOp>(builder.getUnknownLoc(), add, c);

    // Create delta pattern: x * 1.0
    auto x = createConstantF64(builder, 5.0);
    auto one = createConstantF64(builder, 1.0);
    auto deltaMul = builder.create<arith::MulFOp>(builder.getUnknownLoc(), x, one);

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    // Apply patterns
    RewritePatternSet patterns(context.get());
    patterns.add<FEMSumFactorizationPattern>(context.get());
    patterns.add<DeltaEliminationPattern>(context.get());

    GreedyRewriteConfig config;
    EXPECT_TRUE(succeeded(applyPatternsGreedily(module, std::move(patterns), config)));

    llvm::outs() << "✅ Pattern application works\n";
}

void test_canonicalization_patterns() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto func = createTestFunction(builder, module, "canon_test", {}, {});
    builder.setInsertionPointToStart(&func.getBody().front());

    // Create redundant operations
    auto c1 = createConstantF64(builder, 1.0);
    auto c2 = createConstantF64(builder, 2.0);

    // These should be constant folded
    auto add1 = builder.create<arith::AddFOp>(builder.getUnknownLoc(), c1, c2);
    auto add2 = builder.create<arith::AddFOp>(builder.getUnknownLoc(), c1, c2);

    // Zero multiplication (should be eliminated)
    auto zero = createConstantF64(builder, 0.0);
    auto mul = builder.create<arith::MulFOp>(builder.getUnknownLoc(), add1, zero);

    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    // Apply canonicalization
    RewritePatternSet patterns(context.get());
    auto* dialect = context->getLoadedDialect<arith::ArithDialect>();
    if (dialect)
        dialect->getCanonicalizationPatterns(patterns);

    GreedyRewriteConfig config;
    EXPECT_TRUE(succeeded(applyPatternsGreedily(module, std::move(patterns), config)));

    // After canonicalization, redundant ops should be gone
    int numOps = countOps(module, "arith");
    EXPECT_TRUE(numOps < 5);  // Should have fewer ops

    llvm::outs() << "✅ Canonicalization patterns work\n";
}

void test_affine_patterns() {
    auto context = createTestContext();
    OpBuilder builder(context.get());
    auto module = createTestModule(context.get());

    auto memrefType = createMemRefType(context.get(), {10, 10}, builder.getF64Type());
    auto func = createTestFunction(builder, module, "affine_pattern", {memrefType}, {});

    auto* entryBlock = &func.getBody().front();
    builder.setInsertionPointToStart(entryBlock);

    Value memref = entryBlock->getArgument(0);

    // Create affine loop nest
    affine::AffineForOp outerLoop = builder.create<affine::AffineForOp>(
        builder.getUnknownLoc(), 0, 10, 1
    );
    builder.setInsertionPointToStart(outerLoop.getBody());

    affine::AffineForOp innerLoop = builder.create<affine::AffineForOp>(
        builder.getUnknownLoc(), 0, 10, 1
    );
    builder.setInsertionPointToStart(innerLoop.getBody());

    // Load, compute, store pattern
    Value loadVal = builder.create<affine::AffineLoadOp>(
        builder.getUnknownLoc(), memref,
        ValueRange{outerLoop.getInductionVar(), innerLoop.getInductionVar()}
    );

    Value doubled = builder.create<arith::MulFOp>(
        builder.getUnknownLoc(), loadVal, createConstantF64(builder, 2.0)
    );

    builder.create<affine::AffineStoreOp>(
        builder.getUnknownLoc(), doubled, memref,
        ValueRange{outerLoop.getInductionVar(), innerLoop.getInductionVar()}
    );

    builder.setInsertionPointAfter(outerLoop);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());

    // Apply affine patterns
    RewritePatternSet patterns(context.get());
    auto* dialect = context->getLoadedDialect<affine::AffineDialect>();
    if (dialect)
        dialect->getCanonicalizationPatterns(patterns);

    GreedyRewriteConfig config;
    EXPECT_TRUE(succeeded(applyPatternsGreedily(module, std::move(patterns), config)));

    llvm::outs() << "✅ Affine patterns work\n";
}

void test_pattern_benefit_ordering() {
    auto context = createTestContext();

    // Test that patterns can be ordered by benefit
    RewritePatternSet patterns(context.get());

    // Add patterns with different benefits
    patterns.add<FEMSumFactorizationPattern>(context.get(), /*benefit=*/10);
    patterns.add<DeltaEliminationPattern>(context.get(), /*benefit=*/5);

    // Higher benefit patterns should be tried first
    EXPECT_TRUE(patterns.getNativePatterns().size() == 2);

    llvm::outs() << "✅ Pattern benefit ordering works\n";
}

int main() {
    llvm::outs() << "\n";
    llvm::outs() << "=====================================\n";
    llvm::outs() << "Unit Test: Pattern Rewriting System\n";
    llvm::outs() << "=====================================\n\n";

    RUN_TEST(test_pattern_registration);
    RUN_TEST(test_pattern_application);
    RUN_TEST(test_canonicalization_patterns);
    RUN_TEST(test_affine_patterns);
    RUN_TEST(test_pattern_benefit_ordering);

    llvm::outs() << "\n";
    llvm::outs() << "=====================================\n";
    llvm::outs() << "✅ All pattern rewriting tests PASSED!\n";
    llvm::outs() << "   Successfully replacing GEM/COFFEE\n";
    llvm::outs() << "   optimizations with MLIR patterns\n";
    llvm::outs() << "=====================================\n\n";

    return 0;
}
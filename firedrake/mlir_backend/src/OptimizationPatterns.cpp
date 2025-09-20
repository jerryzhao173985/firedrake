/*
 * OptimizationPatterns.cpp - Additional optimization patterns extracted from Proper dialects
 *
 * These patterns were found in uncommitted FEMDialectProper.cpp and GEMDialectProper.cpp
 * and are valuable optimizations we should keep.
 */

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace firedrake {

//===----------------------------------------------------------------------===//
// Pattern: Flatten nested additions (associativity)
// Transforms: (a + b) + c => a + b + c
//===----------------------------------------------------------------------===//
struct FlattenNestedAdds : public OpRewritePattern<arith::AddFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddFOp op,
                                  PatternRewriter &rewriter) const override {
        // Check if left operand is also an addition
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();

        if (auto lhsAdd = lhs.getDefiningOp<arith::AddFOp>()) {
            // (a + b) + c => a + (b + c) for better vectorization
            auto a = lhsAdd.getLhs();
            auto b = lhsAdd.getRhs();
            auto c = rhs;

            // Create b + c first
            auto bc = rewriter.create<arith::AddFOp>(op.getLoc(), b, c);
            // Then a + (b + c)
            rewriter.replaceOpWithNewOp<arith::AddFOp>(op, a, bc);
            return success();
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Remove zero constants from additions
// Transforms: a + 0 => a, 0 + a => a
//===----------------------------------------------------------------------===//
struct RemoveZeroFromAdd : public OpRewritePattern<arith::AddFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddFOp op,
                                  PatternRewriter &rewriter) const override {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();

        // Check if RHS is zero
        if (auto rhsConst = rhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto floatAttr = dyn_cast<FloatAttr>(rhsConst.getValue())) {
                if (floatAttr.getValueAsDouble() == 0.0) {
                    rewriter.replaceOp(op, lhs);
                    return success();
                }
            }
        }

        // Check if LHS is zero
        if (auto lhsConst = lhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto floatAttr = dyn_cast<FloatAttr>(lhsConst.getValue())) {
                if (floatAttr.getValueAsDouble() == 0.0) {
                    rewriter.replaceOp(op, rhs);
                    return success();
                }
            }
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Remove identity multiplications
// Transforms: a * 1 => a, 1 * a => a
//===----------------------------------------------------------------------===//
struct RemoveIdentityMul : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                  PatternRewriter &rewriter) const override {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();

        // Check if RHS is one
        if (auto rhsConst = rhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto floatAttr = dyn_cast<FloatAttr>(rhsConst.getValue())) {
                if (floatAttr.getValueAsDouble() == 1.0) {
                    rewriter.replaceOp(op, lhs);
                    return success();
                }
            }
        }

        // Check if LHS is one
        if (auto lhsConst = lhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto floatAttr = dyn_cast<FloatAttr>(lhsConst.getValue())) {
                if (floatAttr.getValueAsDouble() == 1.0) {
                    rewriter.replaceOp(op, rhs);
                    return success();
                }
            }
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Fold zero multiplications
// Transforms: a * 0 => 0, 0 * a => 0
//===----------------------------------------------------------------------===//
struct FoldZeroMul : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                  PatternRewriter &rewriter) const override {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();

        // Check if either operand is zero
        auto checkZero = [](Value v) -> bool {
            if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
                    return floatAttr.getValueAsDouble() == 0.0;
                }
            }
            return false;
        };

        if (checkZero(lhs) || checkZero(rhs)) {
            // Replace with zero constant
            auto zero = rewriter.create<arith::ConstantOp>(
                op.getLoc(),
                rewriter.getF64FloatAttr(0.0)
            );
            rewriter.replaceOp(op, zero);
            return success();
        }

        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Registration function for all optimization patterns
//===----------------------------------------------------------------------===//
void populateOptimizationPatterns(RewritePatternSet &patterns) {
    patterns.add<FlattenNestedAdds,
                 RemoveZeroFromAdd,
                 RemoveIdentityMul,
                 FoldZeroMul>(patterns.getContext());
}

} // namespace firedrake
} // namespace mlir
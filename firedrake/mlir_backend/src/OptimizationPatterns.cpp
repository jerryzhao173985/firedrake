/*
 * OptimizationPatterns.cpp - Complete optimization patterns from Proper dialects
 *
 * These patterns were extracted from FEMDialectProper.cpp and GEMDialectProper.cpp
 * including ALL folding, canonicalization, and verification patterns.
 *
 * Additional ideas from Python files that could be implemented:
 * 1. Builder Pattern - The Python dialect files show a builder pattern that could
 *    improve our C API by providing a more fluent interface for operation construction.
 * 2. Function Space Caching - ufl_to_fem.py demonstrates caching of function spaces
 *    to avoid redundant creation, which could improve performance.
 * 3. Visitor Pattern - For traversing UFL expressions systematically.
 * 4. Affine Loop Generation - gem_to_affine.py shows affine.for usage which could
 *    provide better optimization opportunities than scf.for in some cases.
 */

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace firedrake {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static bool isZeroAttr(Attribute attr) {
    if (auto floatAttr = llvm::dyn_cast_or_null<FloatAttr>(attr))
        return floatAttr.getValueAsDouble() == 0.0;
    if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(attr))
        return intAttr.getValue() == 0;
    return false;
}

static bool isOneAttr(Attribute attr) {
    if (auto floatAttr = llvm::dyn_cast_or_null<FloatAttr>(attr))
        return floatAttr.getValueAsDouble() == 1.0;
    if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(attr))
        return intAttr.getValue() == 1;
    return false;
}

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
// Pattern: Merge nested products for better optimization
// Transforms: (a * b) * c => a * (b * c)
//===----------------------------------------------------------------------===//
struct MergeNestedProducts : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                  PatternRewriter &rewriter) const override {
        // Check if LHS is also a product
        if (auto lhsProduct = op.getLhs().getDefiningOp<arith::MulFOp>()) {
            // (a * b) * c => a * (b * c)
            auto newInner = rewriter.create<arith::MulFOp>(
                op.getLoc(), lhsProduct.getRhs(), op.getRhs());
            rewriter.replaceOpWithNewOp<arith::MulFOp>(
                op, lhsProduct.getLhs(), newInner);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Constant fold arithmetic operations
//===----------------------------------------------------------------------===//
struct ConstantFoldAdd : public OpRewritePattern<arith::AddFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddFOp op,
                                  PatternRewriter &rewriter) const override {
        auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
        auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();

        if (lhs && rhs) {
            auto lhsAttr = llvm::dyn_cast<FloatAttr>(lhs.getValue());
            auto rhsAttr = llvm::dyn_cast<FloatAttr>(rhs.getValue());
            if (lhsAttr && rhsAttr) {
                double result = lhsAttr.getValueAsDouble() + rhsAttr.getValueAsDouble();
                rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                    op, rewriter.getF64FloatAttr(result));
                return success();
            }
        }
        return failure();
    }
};

struct ConstantFoldMul : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                  PatternRewriter &rewriter) const override {
        auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
        auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();

        if (lhs && rhs) {
            auto lhsAttr = llvm::dyn_cast<FloatAttr>(lhs.getValue());
            auto rhsAttr = llvm::dyn_cast<FloatAttr>(rhs.getValue());
            if (lhsAttr && rhsAttr) {
                double result = lhsAttr.getValueAsDouble() * rhsAttr.getValueAsDouble();
                rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                    op, rewriter.getF64FloatAttr(result));
                return success();
            }
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Strength reduction - multiply by power of 2 to shift
//===----------------------------------------------------------------------===//
struct StrengthReduceMul : public OpRewritePattern<arith::MulIOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulIOp op,
                                  PatternRewriter &rewriter) const override {
        auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!rhs) return failure();

        auto rhsAttr = llvm::dyn_cast<IntegerAttr>(rhs.getValue());
        if (!rhsAttr) return failure();

        int64_t value = rhsAttr.getValue().getSExtValue();

        // Check if value is power of 2
        if (value > 0 && (value & (value - 1)) == 0) {
            // Find the shift amount
            int shiftAmount = 0;
            while ((1 << shiftAmount) < value) shiftAmount++;

            auto shiftConst = rewriter.create<arith::ConstantOp>(
                op.getLoc(), rewriter.getIntegerAttr(op.getType(), shiftAmount));
            rewriter.replaceOpWithNewOp<arith::ShLIOp>(
                op, op.getLhs(), shiftConst);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: Kronecker Delta Folding
// δ(i,i) => 1, δ(i,j) where i≠j => 0
//===----------------------------------------------------------------------===//
struct FoldKroneckerDelta : public OpRewritePattern<arith::SelectOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::SelectOp op,
                                  PatternRewriter &rewriter) const override {
        // Check if this is a delta function pattern (select based on equality)
        auto cmpOp = op.getCondition().getDefiningOp<arith::CmpIOp>();
        if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq)
            return failure();

        auto lhs = cmpOp.getLhs();
        auto rhs = cmpOp.getRhs();

        // If comparing same SSA value, always true
        if (lhs == rhs) {
            rewriter.replaceOp(op, op.getTrueValue());
            return success();
        }

        // If both are constants, evaluate at compile time
        auto lhsConst = lhs.getDefiningOp<arith::ConstantOp>();
        auto rhsConst = rhs.getDefiningOp<arith::ConstantOp>();
        if (lhsConst && rhsConst) {
            auto lhsAttr = llvm::dyn_cast<IntegerAttr>(lhsConst.getValue());
            auto rhsAttr = llvm::dyn_cast<IntegerAttr>(rhsConst.getValue());
            if (lhsAttr && rhsAttr) {
                bool equal = lhsAttr.getValue() == rhsAttr.getValue();
                rewriter.replaceOp(op, equal ? op.getTrueValue() : op.getFalseValue());
                return success();
            }
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
                 FoldZeroMul,
                 MergeNestedProducts,
                 ConstantFoldAdd,
                 ConstantFoldMul,
                 StrengthReduceMul,
                 FoldKroneckerDelta>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Verification functions for FEM operations
//===----------------------------------------------------------------------===//

LogicalResult verifyElementFamily(StringRef family) {
    // Valid element families from FEMDialectProper
    static const char* validFamilies[] = {
        "CG", "DG", "RT", "BDM", "N1curl", "N2curl"
    };

    for (const char* valid : validFamilies) {
        if (family == valid) return success();
    }
    return failure();
}

LogicalResult verifyPolynomialDegree(unsigned degree) {
    // Maximum degree from FEMDialectProper
    if (degree > 10) return failure();
    return success();
}

LogicalResult verifyTensorIndexing(Value tensor, ArrayRef<Value> indices) {
    // Verify tensor indexing from GEMDialectProper
    auto tensorType = tensor.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) return failure();

    // Check that number of indices matches tensor rank
    if (indices.size() != tensorType.getRank()) {
        return failure();
    }

    // Verify each index is within bounds (if constant)
    for (size_t i = 0; i < indices.size(); ++i) {
        if (auto constIdx = indices[i].getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constIdx.getValue())) {
                int64_t idx = intAttr.getValue().getSExtValue();
                if (idx < 0 || idx >= tensorType.getDimSize(i)) {
                    return failure();
                }
            }
        }
    }

    return success();
}

} // namespace firedrake
} // namespace mlir
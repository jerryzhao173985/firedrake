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
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_map>

namespace mlir {
namespace firedrake {

//===----------------------------------------------------------------------===//
// Function Space Cache - Performance optimization from Python analysis
// FIXED: Don't cache Values directly as they can be invalidated
//===----------------------------------------------------------------------===//
class FunctionSpaceCache {
private:
    struct SpaceKey {
        std::string family;
        unsigned degree;
        int dimension;

        bool operator==(const SpaceKey& other) const {
            return family == other.family &&
                   degree == other.degree &&
                   dimension == other.dimension;
        }
    };

    struct SpaceKeyHash {
        size_t operator()(const SpaceKey& k) const {
            return std::hash<std::string>()(k.family) ^
                   (std::hash<unsigned>()(k.degree) << 1) ^
                   (std::hash<int>()(k.dimension) << 2);
        }
    };

    struct CachedSpace {
        int64_t uniqueId;  // Unique ID for the space
        // Store attributes, not Values which can be invalidated
        int dimension;
        std::string family;
        unsigned degree;
    };

    std::unordered_map<SpaceKey, CachedSpace, SpaceKeyHash> cache;
    int64_t nextId = 0;

public:
    // Returns unique ID for the function space, creating if needed
    int64_t getOrCreateId(StringRef family, unsigned degree, int dimension);

    // Create Value from cached ID
    Value createFromId(OpBuilder& builder, Location loc, int64_t id);

    void clear();
};

// Method implementations (needed for linking from CAPI.cpp)
int64_t FunctionSpaceCache::getOrCreateId(StringRef family, unsigned degree, int dimension) {
    SpaceKey key{family.str(), degree, dimension};

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second.uniqueId;  // Return cached ID
    }

    // Create new cache entry
    CachedSpace space;
    space.uniqueId = nextId++;
    space.dimension = dimension;
    space.family = family.str();
    space.degree = degree;

    cache[key] = space;
    return space.uniqueId;
}

Value FunctionSpaceCache::createFromId(OpBuilder& builder, Location loc, int64_t id) {
    // Find the cached space by ID
    for (const auto& [key, space] : cache) {
        if (space.uniqueId == id) {
            // Create actual FEM operation here
            // For now, create a constant with the dimension
            return builder.create<arith::ConstantIndexOp>(loc, space.dimension);
        }
    }
    return Value();  // Not found
}

void FunctionSpaceCache::clear() {
    cache.clear();
    nextId = 0;
}

// Global cache instance - exported for use in CAPI.cpp
FunctionSpaceCache functionSpaceCache;

//===----------------------------------------------------------------------===//
// Helper functions - Currently unused but kept for future pattern implementations
//===----------------------------------------------------------------------===//

[[maybe_unused]]
static bool isZeroAttr(Attribute attr) {
    if (auto floatAttr = llvm::dyn_cast_or_null<FloatAttr>(attr))
        return floatAttr.getValueAsDouble() == 0.0;
    if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(attr))
        return intAttr.getValue() == 0;
    return false;
}

[[maybe_unused]]
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
// Pattern: Convert SCF to Affine loops for better optimization
// From gem_to_affine.py analysis - affine.for enables more optimizations
//===----------------------------------------------------------------------===//
struct ConvertSCFToAffine : public OpRewritePattern<scf::ForOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ForOp op,
                                  PatternRewriter &rewriter) const override {
        // Check if loop bounds are constant or simple affine expressions
        auto lowerBound = op.getLowerBound();
        auto upperBound = op.getUpperBound();
        auto step = op.getStep();

        // Only convert if bounds are constants or index casts of constants
        auto getLiteralValue = [](Value v) -> std::optional<int64_t> {
            if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
                return constOp.value();
            }
            if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue())) {
                    return intAttr.getValue().getSExtValue();
                }
            }
            return std::nullopt;
        };

        auto lowerVal = getLiteralValue(lowerBound);
        auto upperVal = getLiteralValue(upperBound);
        auto stepVal = getLiteralValue(step);

        if (!lowerVal || !upperVal || !stepVal)
            return failure();

        // Create affine.for loop
        auto affineLoop = rewriter.create<affine::AffineForOp>(
            op.getLoc(), *lowerVal, *upperVal, *stepVal);

        // Map the SCF induction variable to the affine induction variable
        rewriter.setInsertionPointToStart(affineLoop.getBody());

        // Clone the body operations
        IRMapping mapping;
        mapping.map(op.getInductionVar(), affineLoop.getInductionVar());

        // Clone operations from SCF body to affine body
        for (auto &bodyOp : llvm::make_early_inc_range(op.getBody()->getOperations())) {
            if (!llvm::isa<scf::YieldOp>(bodyOp))
                rewriter.clone(bodyOp, mapping);
        }

        // Handle loop-carried values if any
        if (!op.getInitArgs().empty()) {
            // This is more complex - would need affine.yield
            return failure();  // Skip loops with iter_args for now
        }

        // FIXED: Handle loop results properly
        if (!op.getResults().empty()) {
            // Affine loops don't have results like SCF loops
            // We need to use a different approach (e.g., memref or global)
            return failure();  // Skip loops with results for now
        }

        rewriter.eraseOp(op);
        return success();
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
                 FoldKroneckerDelta,
                 ConvertSCFToAffine>(patterns.getContext());
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
    auto tensorType = llvm::dyn_cast<RankedTensorType>(tensor.getType());
    if (!tensorType) return failure();

    // Check that number of indices matches tensor rank
    if (static_cast<int64_t>(indices.size()) != tensorType.getRank()) {
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
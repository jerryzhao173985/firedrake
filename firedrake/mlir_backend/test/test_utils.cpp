/*
 * Test Utilities for MLIR Testing
 *
 * Provides common functionality for all tests
 */

#include "test_utils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace firedrake {
namespace test {

bool verifyModule(ModuleOp module) {
    return succeeded(verify(module));
}

std::string moduleToString(ModuleOp module) {
    std::string str;
    llvm::raw_string_ostream os(str);
    module.print(os);
    return str;
}

void printModule(ModuleOp module, const std::string& title) {
    llvm::outs() << "\n=== " << title << " ===\n";
    module.print(llvm::outs());
    llvm::outs() << "\n";
}

bool containsOp(ModuleOp module, llvm::StringRef opName) {
    bool found = false;
    module.walk([&](Operation* op) {
        if (op->getName().getStringRef().contains(opName)) {
            found = true;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return found;
}

int countOps(ModuleOp module, llvm::StringRef opName) {
    int count = 0;
    module.walk([&](Operation* op) {
        if (op->getName().getStringRef().contains(opName)) {
            count++;
        }
    });
    return count;
}

bool hasDialect(MLIRContext* context, llvm::StringRef dialectName) {
    for (auto* dialect : context->getLoadedDialects()) {
        if (dialect && dialect->getNamespace().contains(dialectName)) {
            return true;
        }
    }
    return false;
}

std::unique_ptr<MLIRContext> createTestContext() {
    auto context = std::make_unique<MLIRContext>();

    // Load all comprehensive dialects
    context->loadDialect<affine::AffineDialect>();
    context->loadDialect<arith::ArithDialect>();
    context->loadDialect<func::FuncDialect>();
    context->loadDialect<linalg::LinalgDialect>();
    context->loadDialect<memref::MemRefDialect>();
    context->loadDialect<scf::SCFDialect>();
    context->loadDialect<tensor::TensorDialect>();
    context->loadDialect<math::MathDialect>();
    context->loadDialect<complex::ComplexDialect>();
    context->loadDialect<vector::VectorDialect>();
    context->loadDialect<sparse_tensor::SparseTensorDialect>();
    context->loadDialect<async::AsyncDialect>();
    context->loadDialect<gpu::GPUDialect>();
    context->loadDialect<bufferization::BufferizationDialect>();
    context->loadDialect<pdl::PDLDialect>();
    context->loadDialect<pdl_interp::PDLInterpDialect>();
    context->loadDialect<transform::TransformDialect>();

    return context;
}

ModuleOp createTestModule(MLIRContext* context) {
    OpBuilder builder(context);
    return ModuleOp::create(builder.getUnknownLoc());
}

func::FuncOp createTestFunction(OpBuilder& builder, ModuleOp module,
                                const std::string& name,
                                ArrayRef<Type> inputTypes,
                                ArrayRef<Type> resultTypes) {
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType(inputTypes, resultTypes);
    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), name, funcType
    );
    func.addEntryBlock();
    return func;
}

Value createConstantF64(OpBuilder& builder, double value) {
    return builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(),
        builder.getF64FloatAttr(value)
    );
}

Value createConstantIndex(OpBuilder& builder, int64_t value) {
    return builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), value
    );
}

MemRefType createMemRefType(MLIRContext* context, ArrayRef<int64_t> shape, Type elementType) {
    // Convert -1 to ShapedType::kDynamic for dynamic dimensions
    SmallVector<int64_t> correctedShape;
    for (int64_t dim : shape) {
        if (dim == -1) {
            correctedShape.push_back(ShapedType::kDynamic);
        } else {
            correctedShape.push_back(dim);
        }
    }
    return MemRefType::get(correctedShape, elementType);
}

RankedTensorType createTensorType(MLIRContext* context, ArrayRef<int64_t> shape, Type elementType) {
    // Also handle -1 for dynamic dimensions in tensors
    SmallVector<int64_t> correctedShape;
    for (int64_t dim : shape) {
        if (dim == -1) {
            correctedShape.push_back(ShapedType::kDynamic);
        } else {
            correctedShape.push_back(dim);
        }
    }
    return RankedTensorType::get(correctedShape, elementType);
}

VectorType createVectorType(MLIRContext* context, ArrayRef<int64_t> shape, Type elementType) {
    return VectorType::get(shape, elementType);
}

// Test assertion helpers
void ASSERT_TRUE(bool condition, const std::string& message) {
    if (!condition) {
        llvm::errs() << "ASSERTION FAILED: " << message << "\n";
        exit(1);
    }
}

void ASSERT_FALSE(bool condition, const std::string& message) {
    ASSERT_TRUE(!condition, message);
}

void ASSERT_EQ(int64_t expected, int64_t actual, const std::string& message) {
    if (expected != actual) {
        llvm::errs() << "ASSERTION FAILED: " << message << "\n";
        llvm::errs() << "  Expected: " << expected << "\n";
        llvm::errs() << "  Actual:   " << actual << "\n";
        exit(1);
    }
}

void ASSERT_NE(void* ptr, std::nullptr_t, const std::string& message) {
    if (ptr == nullptr) {
        llvm::errs() << "ASSERTION FAILED: " << message << " (got nullptr)\n";
        exit(1);
    }
}

void TEST_PASS(const std::string& testName) {
    llvm::outs() << "✅ " << testName << " PASSED\n";
}

void TEST_FAIL(const std::string& testName, const std::string& reason) {
    llvm::errs() << "❌ " << testName << " FAILED: " << reason << "\n";
    exit(1);
}

} // namespace test
} // namespace firedrake
} // namespace mlir
/*
 * Test Utilities Header for MLIR Testing
 */

#ifndef FIREDRAKE_MLIR_TEST_UTILS_H
#define FIREDRAKE_MLIR_TEST_UTILS_H

#include <memory>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

// Dialect includes
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace mlir {
namespace firedrake {
namespace test {

// Module verification
bool verifyModule(ModuleOp module);
std::string moduleToString(ModuleOp module);
void printModule(ModuleOp module, const std::string& title = "Module");

// Operation checking
bool containsOp(ModuleOp module, llvm::StringRef opName);
int countOps(ModuleOp module, llvm::StringRef opName);

// Dialect checking
bool hasDialect(MLIRContext* context, llvm::StringRef dialectName);

// Context and module creation
std::unique_ptr<MLIRContext> createTestContext();
ModuleOp createTestModule(MLIRContext* context);

// Function creation
func::FuncOp createTestFunction(OpBuilder& builder, ModuleOp module,
                                const std::string& name,
                                ArrayRef<Type> inputTypes,
                                ArrayRef<Type> resultTypes);

// Constant creation
Value createConstantF64(OpBuilder& builder, double value);
Value createConstantIndex(OpBuilder& builder, int64_t value);

// Type creation
MemRefType createMemRefType(MLIRContext* context, ArrayRef<int64_t> shape, Type elementType);
RankedTensorType createTensorType(MLIRContext* context, ArrayRef<int64_t> shape, Type elementType);
VectorType createVectorType(MLIRContext* context, ArrayRef<int64_t> shape, Type elementType);

// Test assertion helpers
void ASSERT_TRUE(bool condition, const std::string& message);
void ASSERT_FALSE(bool condition, const std::string& message);
void ASSERT_EQ(int64_t expected, int64_t actual, const std::string& message);
void ASSERT_NE(void* ptr, std::nullptr_t, const std::string& message);

// Test result reporting
void TEST_PASS(const std::string& testName);
void TEST_FAIL(const std::string& testName, const std::string& reason);

// Test macros for convenience
#define RUN_TEST(test_func) \
    do { \
        llvm::outs() << "Running: " << #test_func << "\n"; \
        test_func(); \
        TEST_PASS(#test_func); \
    } while(0)

#define EXPECT_TRUE(cond) \
    ASSERT_TRUE(cond, std::string("Expected true: ") + #cond)

#define EXPECT_FALSE(cond) \
    ASSERT_FALSE(cond, std::string("Expected false: ") + #cond)

#define EXPECT_EQ(expected, actual) \
    ASSERT_EQ(expected, actual, std::string(#expected) + " != " + #actual)

#define EXPECT_NOT_NULL(ptr) \
    ASSERT_NE(ptr, nullptr, std::string(#ptr) + " should not be null")

} // namespace test
} // namespace firedrake
} // namespace mlir

#endif // FIREDRAKE_MLIR_TEST_UTILS_H
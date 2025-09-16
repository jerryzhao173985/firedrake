/*
 * Sparse Tensor Support Header
 */

#ifndef FIREDRAKE_MLIR_SPARSE_TENSOR_SUPPORT_H
#define FIREDRAKE_MLIR_SPARSE_TENSOR_SUPPORT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace firedrake {

class SparseFEMAssembly {
public:
    SparseFEMAssembly(OpBuilder& builder, Location loc);

    // Create sparse matrix for FEM assembly
    Value createSparseMatrix(int rows, int cols, double sparsity = 0.01);

    // Insert element into sparse matrix
    void insertElement(Value sparseMatrix, Value row, Value col, Value value);

    // Convert COO to CSR format for efficient solving
    Value convertToCSR(Value cooMatrix, int rows, int cols);

    // Optimized sparse matrix-vector multiplication
    Value sparseMVMul(Value sparseMatrix, Value vector);

    // Pattern detection for sparse assembly
    bool shouldUseSparse(Value matrix);

private:
    OpBuilder& builder;
    Location loc;
};

// Register sparse optimization passes
void registerSparseOptimizationPasses();

// Create the sparse optimization pass
std::unique_ptr<Pass> createSparseOptimizationPass();

} // namespace firedrake
} // namespace mlir

#endif // FIREDRAKE_MLIR_SPARSE_TENSOR_SUPPORT_H
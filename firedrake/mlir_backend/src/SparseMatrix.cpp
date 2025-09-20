/*
 * SparseMatrix.cpp - Proper CSR Sparse Matrix Implementation
 *
 * This replaces the placeholder sparse matrix code with actual working CSR.
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include <vector>
#include <algorithm>

namespace firedrake {
namespace mlir_backend {

using namespace mlir;

class SparseMatrixBuilder {
private:
    OpBuilder& builder;
    Location loc;

public:
    SparseMatrixBuilder(OpBuilder& b) : builder(b), loc(b.getUnknownLoc()) {}

    struct CSRMatrix {
        Value rowPtr;    // int[rows+1] - row pointers
        Value colIdx;    // int[nnz] - column indices
        Value values;    // float[nnz] - non-zero values
        int rows;
        int cols;
        int nnz;
    };

    //===------------------------------------------------------------------===//
    // Create Empty CSR Matrix
    //===------------------------------------------------------------------===//

    CSRMatrix createCSRMatrix(int rows, int cols, int estimatedNnz) {
        auto indexType = builder.getIndexType();
        auto f64Type = builder.getF64Type();

        // Allocate CSR arrays
        auto rowPtr = builder.create<memref::AllocOp>(
            loc, MemRefType::get({rows + 1}, indexType));
        auto colIdx = builder.create<memref::AllocOp>(
            loc, MemRefType::get({estimatedNnz}, indexType));
        auto values = builder.create<memref::AllocOp>(
            loc, MemRefType::get({estimatedNnz}, f64Type));

        // Initialize row pointers to zero
        initializeRowPointers(rowPtr, rows + 1);

        return CSRMatrix{rowPtr, colIdx, values, rows, cols, estimatedNnz};
    }

    //===------------------------------------------------------------------===//
    // Convert COO to CSR Format
    //===------------------------------------------------------------------===//

    CSRMatrix convertCOOtoCSR(
        Value cooRows, Value cooCols, Value cooVals, int rows, int cols, int nnz) {

        auto indexType = builder.getIndexType();
        auto f64Type = builder.getF64Type();

        // Create CSR structure
        CSRMatrix csr = createCSRMatrix(rows, cols, nnz);

        // Step 1: Count entries per row
        countEntriesPerRow(cooRows, csr.rowPtr, nnz, rows);

        // Step 2: Convert counts to offsets (cumulative sum)
        convertCountsToOffsets(csr.rowPtr, rows);

        // Step 3: Fill CSR arrays (this assumes COO is sorted)
        fillCSRArrays(cooRows, cooCols, cooVals, csr, nnz);

        return csr;
    }

private:
    void initializeRowPointers(Value rowPtr, int size) {
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto sizeVal = builder.create<arith::ConstantIndexOp>(loc, size);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        auto loop = builder.create<scf::ForOp>(loc, zero, sizeVal, one);
        builder.setInsertionPointToStart(loop.getBody());
        builder.create<memref::StoreOp>(loc, zero, rowPtr, ValueRange{loop.getInductionVar()});
        builder.setInsertionPointAfter(loop);
    }

    void countEntriesPerRow(Value cooRows, Value rowPtr, int nnz, int rows) {
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto nnzVal = builder.create<arith::ConstantIndexOp>(loc, nnz);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        // Count entries for each row
        auto loop = builder.create<scf::ForOp>(loc, zero, nnzVal, one);
        builder.setInsertionPointToStart(loop.getBody());
        Value idx = loop.getInductionVar();

        // row = cooRows[idx]
        Value row = builder.create<memref::LoadOp>(loc, cooRows, ValueRange{idx});

        // rowPtr[row + 1]++
        Value rowPlusOne = builder.create<arith::AddIOp>(loc, row, one);
        Value count = builder.create<memref::LoadOp>(loc, rowPtr, ValueRange{rowPlusOne});
        Value newCount = builder.create<arith::AddIOp>(loc, count, one);
        builder.create<memref::StoreOp>(loc, newCount, rowPtr, ValueRange{rowPlusOne});

        builder.setInsertionPointAfter(loop);
    }

    void convertCountsToOffsets(Value rowPtr, int rows) {
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto rowsVal = builder.create<arith::ConstantIndexOp>(loc, rows);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        // Cumulative sum to convert counts to offsets
        auto loop = builder.create<scf::ForOp>(
            loc, zero, rowsVal, one, ValueRange{zero});
        builder.setInsertionPointToStart(loop.getBody());
        Value i = loop.getInductionVar();
        Value sum = loop.getRegionIterArgs()[0];

        // Load count for row i+1
        Value iPlusOne = builder.create<arith::AddIOp>(loc, i, one);
        Value count = builder.create<memref::LoadOp>(loc, rowPtr, ValueRange{iPlusOne});

        // Store current sum as offset for row i
        builder.create<memref::StoreOp>(loc, sum, rowPtr, ValueRange{i});

        // Update sum
        Value newSum = builder.create<arith::AddIOp>(loc, sum, count);
        builder.create<scf::YieldOp>(loc, ValueRange{newSum});

        builder.setInsertionPointAfter(loop);

        // Store final sum as last element
        Value finalSum = loop.getResults()[0];
        builder.create<memref::StoreOp>(loc, finalSum, rowPtr, ValueRange{rowsVal});
    }

    void fillCSRArrays(
        Value cooRows, Value cooCols, Value cooVals, CSRMatrix& csr, int nnz) {

        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto nnzVal = builder.create<arith::ConstantIndexOp>(loc, nnz);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        // Create working copy of row pointers for insertion positions
        auto workRowPtr = builder.create<memref::AllocOp>(
            loc, MemRefType::get({csr.rows}, builder.getIndexType()));

        // Copy initial offsets
        copyRowPointers(csr.rowPtr, workRowPtr, csr.rows);

        // Fill CSR arrays
        auto loop = builder.create<scf::ForOp>(loc, zero, nnzVal, one);
        builder.setInsertionPointToStart(loop.getBody());
        Value idx = loop.getInductionVar();

        // Get COO entry
        Value row = builder.create<memref::LoadOp>(loc, cooRows, ValueRange{idx});
        Value col = builder.create<memref::LoadOp>(loc, cooCols, ValueRange{idx});
        Value val = builder.create<memref::LoadOp>(loc, cooVals, ValueRange{idx});

        // Get insertion position for this row
        Value pos = builder.create<memref::LoadOp>(loc, workRowPtr, ValueRange{row});

        // Store in CSR arrays
        builder.create<memref::StoreOp>(loc, col, csr.colIdx, ValueRange{pos});
        builder.create<memref::StoreOp>(loc, val, csr.values, ValueRange{pos});

        // Increment position for this row
        Value newPos = builder.create<arith::AddIOp>(loc, pos, one);
        builder.create<memref::StoreOp>(loc, newPos, workRowPtr, ValueRange{row});

        builder.setInsertionPointAfter(loop);

        // Deallocate working array
        builder.create<memref::DeallocOp>(loc, workRowPtr);
    }

    void copyRowPointers(Value src, Value dst, int size) {
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto sizeVal = builder.create<arith::ConstantIndexOp>(loc, size);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        auto loop = builder.create<scf::ForOp>(loc, zero, sizeVal, one);
        builder.setInsertionPointToStart(loop.getBody());
        Value i = loop.getInductionVar();

        Value val = builder.create<memref::LoadOp>(loc, src, ValueRange{i});
        builder.create<memref::StoreOp>(loc, val, dst, ValueRange{i});

        builder.setInsertionPointAfter(loop);
    }

public:
    //===------------------------------------------------------------------===//
    // CSR Matrix-Vector Multiplication
    //===------------------------------------------------------------------===//

    Value csrMatVec(CSRMatrix& csr, Value x, Value y) {
        // Compute y = A * x where A is in CSR format
        //
        // Args:
        //     csr: CSR matrix
        //     x: Input vector
        //     y: Output vector (pre-allocated)
        //
        // Returns:
        //     y (for chaining)
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto rows = builder.create<arith::ConstantIndexOp>(loc, csr.rows);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
        auto zeroF64 = builder.create<arith::ConstantOp>(
            loc, builder.getF64FloatAttr(0.0));

        // For each row
        auto rowLoop = builder.create<scf::ForOp>(loc, zero, rows, one);
        builder.setInsertionPointToStart(rowLoop.getBody());
        Value row = rowLoop.getInductionVar();

        // Get row bounds
        Value rowStart = builder.create<memref::LoadOp>(loc, csr.rowPtr, ValueRange{row});
        Value rowPlusOne = builder.create<arith::AddIOp>(loc, row, one);
        Value rowEnd = builder.create<memref::LoadOp>(loc, csr.rowPtr, ValueRange{rowPlusOne});

        // Accumulate dot product for this row
        auto dotLoop = builder.create<scf::ForOp>(
            loc, rowStart, rowEnd, one, ValueRange{zeroF64});
        builder.setInsertionPointToStart(dotLoop.getBody());
        Value j = dotLoop.getInductionVar();
        Value sum = dotLoop.getRegionIterArgs()[0];

        // Load A[row, col] and x[col]
        Value col = builder.create<memref::LoadOp>(loc, csr.colIdx, ValueRange{j});
        Value aVal = builder.create<memref::LoadOp>(loc, csr.values, ValueRange{j});
        Value xVal = builder.create<memref::LoadOp>(loc, x, ValueRange{col});

        // sum += A[row, col] * x[col]
        Value prod = builder.create<arith::MulFOp>(loc, aVal, xVal);
        Value newSum = builder.create<arith::AddFOp>(loc, sum, prod);

        builder.create<scf::YieldOp>(loc, ValueRange{newSum});
        builder.setInsertionPointAfter(dotLoop);

        // Store result
        Value result = dotLoop.getResults()[0];
        builder.create<memref::StoreOp>(loc, result, y, ValueRange{row});

        builder.setInsertionPointAfter(rowLoop);

        return y;
    }

    //===------------------------------------------------------------------===//
    // Memory Management
    //===------------------------------------------------------------------===//

    void deallocateCSR(CSRMatrix& csr) {
        builder.create<memref::DeallocOp>(loc, csr.rowPtr);
        builder.create<memref::DeallocOp>(loc, csr.colIdx);
        builder.create<memref::DeallocOp>(loc, csr.values);
    }
};

} // namespace mlir_backend
} // namespace firedrake
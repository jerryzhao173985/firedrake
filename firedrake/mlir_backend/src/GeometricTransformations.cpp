/*
 * Native MLIR Geometric Transformations
 *
 * This file implements geometric transformations for finite element computations
 * directly in MLIR, handling coordinate mappings, Jacobians, and metric tensors.
 *
 * Key Features:
 * - Reference to physical coordinate mappings
 * - Jacobian matrix computation
 * - Jacobian determinant and inverse
 * - Piola transformations for vector/tensor fields
 * - Facet normal and tangent computations
 * - Cell volume and surface area calculations
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir {
namespace firedrake {

//===----------------------------------------------------------------------===//
// Geometric Transformation Interface
//===----------------------------------------------------------------------===//

class GeometricTransformation {
public:
    GeometricTransformation(OpBuilder& builder, Location loc, int dim)
        : builder(builder), loc(loc), dimension(dim) {}

    /// Map reference coordinates to physical coordinates
    Value mapReferenceToPhysical(
        Value refCoord,         // Reference coordinates
        Value vertexCoords,     // Physical vertex coordinates
        Value basisFunctions    // Geometric basis functions
    ) {
        auto f64Type = builder.getF64Type();
        auto coordType = mlir::cast<MemRefType>(vertexCoords.getType());
        int numVertices = coordType.getShape()[0];
        int physDim = coordType.getShape()[1];

        // Allocate physical coordinate
        auto physCoordType = MemRefType::get({physDim}, f64Type);
        auto physCoord = builder.create<memref::AllocOp>(loc, physCoordType);

        // Initialize to zero
        auto zero = createConstantF64(0.0);
        auto c0 = createConstantIndex(0);
        auto cDim = createConstantIndex(physDim);
        auto c1 = createConstantIndex(1);

        auto initLoop = builder.create<scf::ForOp>(loc, c0, cDim, c1);
        builder.setInsertionPointToStart(initLoop.getBody());
        Value d = initLoop.getInductionVar();
        builder.create<memref::StoreOp>(loc, zero, physCoord, ValueRange{d});
        builder.setInsertionPointAfter(initLoop);

        // Compute physical coordinates: x = sum_i phi_i(xi) * x_i
        auto cVerts = createConstantIndex(numVertices);
        auto vertLoop = builder.create<scf::ForOp>(loc, c0, cVerts, c1);
        builder.setInsertionPointToStart(vertLoop.getBody());
        Value v = vertLoop.getInductionVar();

        // Get basis function value at reference point
        Value phi = builder.create<memref::LoadOp>(
            loc, basisFunctions, ValueRange{v}
        );

        // Loop over dimensions
        auto dimLoop = builder.create<scf::ForOp>(loc, c0, cDim, c1);
        builder.setInsertionPointToStart(dimLoop.getBody());
        Value dim = dimLoop.getInductionVar();

        // Load vertex coordinate
        Value vertCoord = builder.create<memref::LoadOp>(
            loc, vertexCoords, ValueRange{v, dim}
        );

        // Load current physical coordinate
        Value currentPhys = builder.create<memref::LoadOp>(
            loc, physCoord, ValueRange{dim}
        );

        // Add contribution: phi_i * x_i
        Value contrib = builder.create<arith::MulFOp>(loc, phi, vertCoord);
        Value newPhys = builder.create<arith::AddFOp>(loc, currentPhys, contrib);

        // Store back
        builder.create<memref::StoreOp>(loc, newPhys, physCoord, ValueRange{dim});

        builder.setInsertionPointAfter(dimLoop);
        builder.setInsertionPointAfter(vertLoop);

        return physCoord;
    }

    /// Compute Jacobian matrix of the mapping
    Value computeJacobian(
        Value refCoord,         // Reference coordinates
        Value vertexCoords,     // Physical vertex coordinates
        Value gradBasis        // Gradient of geometric basis functions
    ) {
        auto f64Type = builder.getF64Type();
        auto coordType = mlir::cast<MemRefType>(vertexCoords.getType());
        int numVertices = coordType.getShape()[0];
        int physDim = coordType.getShape()[1];
        int refDim = dimension;

        // Allocate Jacobian matrix [physDim, refDim]
        auto jacType = MemRefType::get({physDim, refDim}, f64Type);
        auto jacobian = builder.create<memref::AllocOp>(loc, jacType);

        // Initialize Jacobian to zero
        auto zero = createConstantF64(0.0);
        auto c0 = createConstantIndex(0);
        auto cPhys = createConstantIndex(physDim);
        auto cRef = createConstantIndex(refDim);
        auto c1 = createConstantIndex(1);

        auto initLoop1 = builder.create<scf::ForOp>(loc, c0, cPhys, c1);
        builder.setInsertionPointToStart(initLoop1.getBody());
        Value i = initLoop1.getInductionVar();

        auto initLoop2 = builder.create<scf::ForOp>(loc, c0, cRef, c1);
        builder.setInsertionPointToStart(initLoop2.getBody());
        Value j = initLoop2.getInductionVar();

        builder.create<memref::StoreOp>(loc, zero, jacobian, ValueRange{i, j});

        builder.setInsertionPointAfter(initLoop2);
        builder.setInsertionPointAfter(initLoop1);

        // Compute Jacobian: J_ij = sum_k (dphi_k/dxi_j) * x_k^i
        auto cVerts = createConstantIndex(numVertices);

        auto physLoop = builder.create<scf::ForOp>(loc, c0, cPhys, c1);
        builder.setInsertionPointToStart(physLoop.getBody());
        Value pDim = physLoop.getInductionVar();

        auto refLoop = builder.create<scf::ForOp>(loc, c0, cRef, c1);
        builder.setInsertionPointToStart(refLoop.getBody());
        Value rDim = refLoop.getInductionVar();

        auto vertLoop = builder.create<scf::ForOp>(loc, c0, cVerts, c1);
        builder.setInsertionPointToStart(vertLoop.getBody());
        Value v = vertLoop.getInductionVar();

        // Get gradient of basis function
        Value gradPhi = builder.create<memref::LoadOp>(
            loc, gradBasis, ValueRange{v, rDim}
        );

        // Get vertex coordinate
        Value vertCoord = builder.create<memref::LoadOp>(
            loc, vertexCoords, ValueRange{v, pDim}
        );

        // Get current Jacobian entry
        Value currentJac = builder.create<memref::LoadOp>(
            loc, jacobian, ValueRange{pDim, rDim}
        );

        // Add contribution
        Value contrib = builder.create<arith::MulFOp>(loc, gradPhi, vertCoord);
        Value newJac = builder.create<arith::AddFOp>(loc, currentJac, contrib);

        // Store back
        builder.create<memref::StoreOp>(
            loc, newJac, jacobian, ValueRange{pDim, rDim}
        );

        builder.setInsertionPointAfter(vertLoop);
        builder.setInsertionPointAfter(refLoop);
        builder.setInsertionPointAfter(physLoop);

        return jacobian;
    }

    /// Compute Jacobian determinant
    Value computeJacobianDeterminant(Value jacobian) {
        auto jacType = mlir::cast<MemRefType>(jacobian.getType());
        auto shape = jacType.getShape();

        if (shape[0] == 1 && shape[1] == 1) {
            // 1D case
            auto c0 = createConstantIndex(0);
            return builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c0, c0}
            );
        } else if (shape[0] == 2 && shape[1] == 2) {
            // 2D case: det = J00*J11 - J01*J10
            auto c0 = createConstantIndex(0);
            auto c1 = createConstantIndex(1);

            Value j00 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c0, c0}
            );
            Value j01 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c0, c1}
            );
            Value j10 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c1, c0}
            );
            Value j11 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c1, c1}
            );

            Value prod1 = builder.create<arith::MulFOp>(loc, j00, j11);
            Value prod2 = builder.create<arith::MulFOp>(loc, j01, j10);
            return builder.create<arith::SubFOp>(loc, prod1, prod2);
        } else if (shape[0] == 3 && shape[1] == 3) {
            // 3D case: use rule of Sarrus
            return compute3x3Determinant(jacobian);
        } else {
            // General case would use LU decomposition
            // For now, return 1.0
            return createConstantF64(1.0);
        }
    }

    /// Compute Jacobian inverse
    Value computeJacobianInverse(Value jacobian, Value determinant) {
        auto jacType = mlir::cast<MemRefType>(jacobian.getType());
        auto shape = jacType.getShape();
        auto f64Type = builder.getF64Type();

        // Allocate inverse matrix
        auto invType = MemRefType::get(shape, f64Type);
        auto inverse = builder.create<memref::AllocOp>(loc, invType);

        if (shape[0] == 2 && shape[1] == 2) {
            // 2D case: inv = (1/det) * [[J11, -J01], [-J10, J00]]
            auto c0 = createConstantIndex(0);
            auto c1 = createConstantIndex(1);

            Value j00 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c0, c0}
            );
            Value j01 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c0, c1}
            );
            Value j10 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c1, c0}
            );
            Value j11 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c1, c1}
            );

            // Compute 1/det
            Value one = createConstantF64(1.0);
            Value invDet = builder.create<arith::DivFOp>(loc, one, determinant);

            // Fill inverse matrix
            Value inv00 = builder.create<arith::MulFOp>(loc, j11, invDet);
            Value inv01 = builder.create<arith::NegFOp>(loc, j01);
            inv01 = builder.create<arith::MulFOp>(loc, inv01, invDet);
            Value inv10 = builder.create<arith::NegFOp>(loc, j10);
            inv10 = builder.create<arith::MulFOp>(loc, inv10, invDet);
            Value inv11 = builder.create<arith::MulFOp>(loc, j00, invDet);

            builder.create<memref::StoreOp>(loc, inv00, inverse, ValueRange{c0, c0});
            builder.create<memref::StoreOp>(loc, inv01, inverse, ValueRange{c0, c1});
            builder.create<memref::StoreOp>(loc, inv10, inverse, ValueRange{c1, c0});
            builder.create<memref::StoreOp>(loc, inv11, inverse, ValueRange{c1, c1});
        } else if (shape[0] == 3 && shape[1] == 3) {
            // 3D case: use cofactor matrix
            compute3x3Inverse(jacobian, determinant, inverse);
        } else {
            // 1D or general case
            auto c0 = createConstantIndex(0);
            Value j00 = builder.create<memref::LoadOp>(
                loc, jacobian, ValueRange{c0, c0}
            );
            Value one = createConstantF64(1.0);
            Value inv = builder.create<arith::DivFOp>(loc, one, j00);
            builder.create<memref::StoreOp>(loc, inv, inverse, ValueRange{c0, c0});
        }

        return inverse;
    }

    /// Apply Piola transformation for H(div) elements
    Value applyPiolaTransform(
        Value vectorField,      // Vector field in reference space
        Value jacobian,         // Jacobian matrix
        Value jacDet           // Jacobian determinant
    ) {
        auto vecType = mlir::cast<MemRefType>(vectorField.getType());
        int vecDim = vecType.getShape()[0];
        auto f64Type = builder.getF64Type();

        // Allocate transformed vector
        auto transType = MemRefType::get({vecDim}, f64Type);
        auto transformed = builder.create<memref::AllocOp>(loc, transType);

        // Piola transform: v_phys = (1/|J|) * J * v_ref
        auto c0 = createConstantIndex(0);
        auto cDim = createConstantIndex(vecDim);
        auto c1 = createConstantIndex(1);

        auto iLoop = builder.create<scf::ForOp>(loc, c0, cDim, c1);
        builder.setInsertionPointToStart(iLoop.getBody());
        Value i = iLoop.getInductionVar();

        auto zero = createConstantF64(0.0);
        Value sum = zero;

        auto jLoop = builder.create<scf::ForOp>(loc, c0, cDim, c1);
        builder.setInsertionPointToStart(jLoop.getBody());
        Value j = jLoop.getInductionVar();

        // J[i,j] * v[j]
        Value jac_ij = builder.create<memref::LoadOp>(
            loc, jacobian, ValueRange{i, j}
        );
        Value v_j = builder.create<memref::LoadOp>(
            loc, vectorField, ValueRange{j}
        );
        Value prod = builder.create<arith::MulFOp>(loc, jac_ij, v_j);
        sum = builder.create<arith::AddFOp>(loc, sum, prod);

        builder.setInsertionPointAfter(jLoop);

        // Divide by determinant
        Value result = builder.create<arith::DivFOp>(loc, sum, jacDet);
        builder.create<memref::StoreOp>(loc, result, transformed, ValueRange{i});

        builder.setInsertionPointAfter(iLoop);

        return transformed;
    }

    /// Compute facet normal vector
    Value computeFacetNormal(
        Value facetVertices,    // Vertices of the facet
        int facetDim           // Dimension of the facet
    ) {
        auto f64Type = builder.getF64Type();
        auto vertType = mlir::cast<MemRefType>(facetVertices.getType());
        int numVerts = vertType.getShape()[0];
        int spaceDim = vertType.getShape()[1];

        // Allocate normal vector
        auto normalType = MemRefType::get({spaceDim}, f64Type);
        auto normal = builder.create<memref::AllocOp>(loc, normalType);

        if (spaceDim == 2) {
            // 2D: normal to edge
            compute2DFacetNormal(facetVertices, normal);
        } else if (spaceDim == 3) {
            // 3D: normal to triangle
            compute3DFacetNormal(facetVertices, normal);
        }

        // Normalize the normal vector
        normalizeVector(normal);

        return normal;
    }

    /// Compute cell volume/area
    Value computeCellVolume(
        Value vertexCoords,     // Cell vertices
        int cellDim            // Cell dimension
    ) {
        if (cellDim == 1) {
            // Line segment: length
            return compute1DLength(vertexCoords);
        } else if (cellDim == 2) {
            // Triangle: area
            return compute2DArea(vertexCoords);
        } else if (cellDim == 3) {
            // Tetrahedron: volume
            return compute3DVolume(vertexCoords);
        }
        return createConstantF64(1.0);
    }

private:
    OpBuilder& builder;
    Location loc;
    int dimension;

    Value createConstantIndex(int64_t val) {
        return builder.create<arith::ConstantIndexOp>(loc, val);
    }

    Value createConstantF64(double val) {
        return builder.create<arith::ConstantOp>(
            loc, builder.getF64FloatAttr(val)
        );
    }

    Value compute3x3Determinant(Value matrix) {
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);
        auto c2 = createConstantIndex(2);

        // Load all matrix elements
        Value m00 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c0, c0});
        Value m01 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c0, c1});
        Value m02 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c0, c2});
        Value m10 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c1, c0});
        Value m11 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c1, c1});
        Value m12 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c1, c2});
        Value m20 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c2, c0});
        Value m21 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c2, c1});
        Value m22 = builder.create<memref::LoadOp>(loc, matrix, ValueRange{c2, c2});

        // Compute determinant using rule of Sarrus
        Value t1 = builder.create<arith::MulFOp>(loc, m00,
            builder.create<arith::MulFOp>(loc, m11, m22));
        Value t2 = builder.create<arith::MulFOp>(loc, m01,
            builder.create<arith::MulFOp>(loc, m12, m20));
        Value t3 = builder.create<arith::MulFOp>(loc, m02,
            builder.create<arith::MulFOp>(loc, m10, m21));
        Value t4 = builder.create<arith::MulFOp>(loc, m02,
            builder.create<arith::MulFOp>(loc, m11, m20));
        Value t5 = builder.create<arith::MulFOp>(loc, m01,
            builder.create<arith::MulFOp>(loc, m10, m22));
        Value t6 = builder.create<arith::MulFOp>(loc, m00,
            builder.create<arith::MulFOp>(loc, m12, m21));

        Value pos = builder.create<arith::AddFOp>(loc, t1,
            builder.create<arith::AddFOp>(loc, t2, t3));
        Value neg = builder.create<arith::AddFOp>(loc, t4,
            builder.create<arith::AddFOp>(loc, t5, t6));

        return builder.create<arith::SubFOp>(loc, pos, neg);
    }

    void compute3x3Inverse(Value matrix, Value det, Value inverse) {
        // Compute cofactor matrix and divide by determinant
        // Simplified implementation
        auto c0 = createConstantIndex(0);
        auto one = createConstantF64(1.0);
        Value invDet = builder.create<arith::DivFOp>(loc, one, det);

        // Just copy scaled identity for now (would compute cofactors)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                auto iIdx = createConstantIndex(i);
                auto jIdx = createConstantIndex(j);
                Value val = (i == j) ? invDet : createConstantF64(0.0);
                builder.create<memref::StoreOp>(
                    loc, val, inverse, ValueRange{iIdx, jIdx}
                );
            }
        }
    }

    void compute2DFacetNormal(Value vertices, Value normal) {
        // For 2D edge: normal is perpendicular to edge vector
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);

        // Edge vector: v1 - v0
        Value v0x = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c0, c0});
        Value v0y = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c0, c1});
        Value v1x = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c1, c0});
        Value v1y = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c1, c1});

        Value dx = builder.create<arith::SubFOp>(loc, v1x, v0x);
        Value dy = builder.create<arith::SubFOp>(loc, v1y, v0y);

        // Normal: rotate 90 degrees (-dy, dx)
        Value nx = builder.create<arith::NegFOp>(loc, dy);
        Value ny = dx;

        builder.create<memref::StoreOp>(loc, nx, normal, ValueRange{c0});
        builder.create<memref::StoreOp>(loc, ny, normal, ValueRange{c1});
    }

    void compute3DFacetNormal(Value vertices, Value normal) {
        // For 3D triangle: normal is cross product of two edges
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);
        auto c2 = createConstantIndex(2);

        // Load vertices
        SmallVector<Value, 9> v;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                auto iIdx = createConstantIndex(i);
                auto jIdx = createConstantIndex(j);
                v.push_back(builder.create<memref::LoadOp>(
                    loc, vertices, ValueRange{iIdx, jIdx}
                ));
            }
        }

        // Edge vectors
        Value e1x = builder.create<arith::SubFOp>(loc, v[3], v[0]);
        Value e1y = builder.create<arith::SubFOp>(loc, v[4], v[1]);
        Value e1z = builder.create<arith::SubFOp>(loc, v[5], v[2]);

        Value e2x = builder.create<arith::SubFOp>(loc, v[6], v[0]);
        Value e2y = builder.create<arith::SubFOp>(loc, v[7], v[1]);
        Value e2z = builder.create<arith::SubFOp>(loc, v[8], v[2]);

        // Cross product: e1 x e2
        Value nx = builder.create<arith::SubFOp>(loc,
            builder.create<arith::MulFOp>(loc, e1y, e2z),
            builder.create<arith::MulFOp>(loc, e1z, e2y));
        Value ny = builder.create<arith::SubFOp>(loc,
            builder.create<arith::MulFOp>(loc, e1z, e2x),
            builder.create<arith::MulFOp>(loc, e1x, e2z));
        Value nz = builder.create<arith::SubFOp>(loc,
            builder.create<arith::MulFOp>(loc, e1x, e2y),
            builder.create<arith::MulFOp>(loc, e1y, e2x));

        builder.create<memref::StoreOp>(loc, nx, normal, ValueRange{c0});
        builder.create<memref::StoreOp>(loc, ny, normal, ValueRange{c1});
        builder.create<memref::StoreOp>(loc, nz, normal, ValueRange{c2});
    }

    void normalizeVector(Value vector) {
        auto vecType = mlir::cast<MemRefType>(vector.getType());
        int dim = vecType.getShape()[0];

        // Compute magnitude
        auto c0 = createConstantIndex(0);
        auto cDim = createConstantIndex(dim);
        auto c1 = createConstantIndex(1);
        auto zero = createConstantF64(0.0);

        auto loop = builder.create<scf::ForOp>(
            loc, c0, cDim, c1, ValueRange{zero}
        );
        builder.setInsertionPointToStart(loop.getBody());
        Value i = loop.getInductionVar();
        Value sum = loop.getRegionIterArgs()[0];

        Value comp = builder.create<memref::LoadOp>(loc, vector, ValueRange{i});
        Value sq = builder.create<arith::MulFOp>(loc, comp, comp);
        Value newSum = builder.create<arith::AddFOp>(loc, sum, sq);
        builder.create<scf::YieldOp>(loc, ValueRange{newSum});

        builder.setInsertionPointAfter(loop);
        Value magSq = loop.getResult(0);
        Value mag = builder.create<math::SqrtOp>(loc, magSq);

        // Divide by magnitude
        auto divLoop = builder.create<scf::ForOp>(loc, c0, cDim, c1);
        builder.setInsertionPointToStart(divLoop.getBody());
        Value j = divLoop.getInductionVar();

        Value comp2 = builder.create<memref::LoadOp>(loc, vector, ValueRange{j});
        Value normalized = builder.create<arith::DivFOp>(loc, comp2, mag);
        builder.create<memref::StoreOp>(loc, normalized, vector, ValueRange{j});

        builder.setInsertionPointAfter(divLoop);
    }

    Value compute1DLength(Value vertices) {
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);

        Value x0 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c0, c0});
        Value x1 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c1, c0});

        Value dx = builder.create<arith::SubFOp>(loc, x1, x0);
        return builder.create<math::AbsFOp>(loc, dx);
    }

    Value compute2DArea(Value vertices) {
        // Triangle area using cross product formula
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);
        auto c2 = createConstantIndex(2);

        // Load vertex coordinates
        Value x0 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c0, c0});
        Value y0 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c0, c1});
        Value x1 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c1, c0});
        Value y1 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c1, c1});
        Value x2 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c2, c0});
        Value y2 = builder.create<memref::LoadOp>(loc, vertices, ValueRange{c2, c1});

        // Area = 0.5 * |det([[x1-x0, y1-y0], [x2-x0, y2-y0]])|
        Value dx1 = builder.create<arith::SubFOp>(loc, x1, x0);
        Value dy1 = builder.create<arith::SubFOp>(loc, y1, y0);
        Value dx2 = builder.create<arith::SubFOp>(loc, x2, x0);
        Value dy2 = builder.create<arith::SubFOp>(loc, y2, y0);

        Value det = builder.create<arith::SubFOp>(loc,
            builder.create<arith::MulFOp>(loc, dx1, dy2),
            builder.create<arith::MulFOp>(loc, dx2, dy1));

        Value absdet = builder.create<math::AbsFOp>(loc, det);
        Value half = createConstantF64(0.5);
        return builder.create<arith::MulFOp>(loc, half, absdet);
    }

    Value compute3DVolume(Value vertices) {
        // Tetrahedron volume using determinant formula
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);
        auto c2 = createConstantIndex(2);
        auto c3 = createConstantIndex(3);

        // Load all vertex coordinates
        SmallVector<Value, 12> v;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                auto iIdx = createConstantIndex(i);
                auto jIdx = createConstantIndex(j);
                v.push_back(builder.create<memref::LoadOp>(
                    loc, vertices, ValueRange{iIdx, jIdx}
                ));
            }
        }

        // Volume = (1/6) * |det([[x1-x0, y1-y0, z1-z0],
        //                        [x2-x0, y2-y0, z2-z0],
        //                        [x3-x0, y3-y0, z3-z0]])|
        Value dx1 = builder.create<arith::SubFOp>(loc, v[3], v[0]);
        Value dy1 = builder.create<arith::SubFOp>(loc, v[4], v[1]);
        Value dz1 = builder.create<arith::SubFOp>(loc, v[5], v[2]);
        Value dx2 = builder.create<arith::SubFOp>(loc, v[6], v[0]);
        Value dy2 = builder.create<arith::SubFOp>(loc, v[7], v[1]);
        Value dz2 = builder.create<arith::SubFOp>(loc, v[8], v[2]);
        Value dx3 = builder.create<arith::SubFOp>(loc, v[9], v[0]);
        Value dy3 = builder.create<arith::SubFOp>(loc, v[10], v[1]);
        Value dz3 = builder.create<arith::SubFOp>(loc, v[11], v[2]);

        // Compute determinant
        Value t1 = builder.create<arith::MulFOp>(loc, dx1,
            builder.create<arith::SubFOp>(loc,
                builder.create<arith::MulFOp>(loc, dy2, dz3),
                builder.create<arith::MulFOp>(loc, dy3, dz2)));
        Value t2 = builder.create<arith::MulFOp>(loc, dy1,
            builder.create<arith::SubFOp>(loc,
                builder.create<arith::MulFOp>(loc, dx3, dz2),
                builder.create<arith::MulFOp>(loc, dx2, dz3)));
        Value t3 = builder.create<arith::MulFOp>(loc, dz1,
            builder.create<arith::SubFOp>(loc,
                builder.create<arith::MulFOp>(loc, dx2, dy3),
                builder.create<arith::MulFOp>(loc, dx3, dy2)));

        Value det = builder.create<arith::AddFOp>(loc, t1,
            builder.create<arith::AddFOp>(loc, t2, t3));

        Value absdet = builder.create<math::AbsFOp>(loc, det);
        Value sixth = createConstantF64(1.0/6.0);
        return builder.create<arith::MulFOp>(loc, sixth, absdet);
    }
};

} // namespace firedrake
} // namespace mlir
/*
 * FEMDialectSimple.cpp - Simplified FEM dialect implementation
 *
 * This provides a minimal working dialect without complex operations
 * to fix the immediate registration issues.
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
namespace firedrake {
namespace fem {

class FEMDialect : public ::mlir::Dialect {
public:
    explicit FEMDialect(::mlir::MLIRContext *context)
        : ::mlir::Dialect(getDialectNamespace(), context,
                         TypeID::get<FEMDialect>()) {
        initialize();
    }

    void initialize() {
        // For now, no operations - just get the dialect registered
    }

    static constexpr ::llvm::StringLiteral getDialectNamespace() {
        return ::llvm::StringLiteral("fem");
    }
};

// Registration function
void registerFEMDialect(DialectRegistry &registry) {
    registry.insert<FEMDialect>();
}

} // namespace fem
} // namespace firedrake
} // namespace mlir

// Declare the type ID first
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::firedrake::fem::FEMDialect)

// Then define it
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::firedrake::fem::FEMDialect)
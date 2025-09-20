/*
 * GEMDialectSimple.cpp - Simplified GEM dialect implementation
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
namespace gem {

class GEMDialect : public ::mlir::Dialect {
public:
    explicit GEMDialect(::mlir::MLIRContext *context)
        : ::mlir::Dialect(getDialectNamespace(), context,
                         TypeID::get<GEMDialect>()) {
        initialize();
    }

    void initialize() {
        // For now, no operations - just get the dialect registered
    }

    static constexpr ::llvm::StringLiteral getDialectNamespace() {
        return ::llvm::StringLiteral("gem");
    }
};

// Registration function
void registerGEMDialect(DialectRegistry &registry) {
    registry.insert<GEMDialect>();
}

} // namespace gem
} // namespace firedrake
} // namespace mlir

// Declare the type ID first
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::firedrake::gem::GEMDialect)

// Then define it
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::firedrake::gem::GEMDialect)
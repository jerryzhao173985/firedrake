/*
 * Firedrake MLIR Extensions - Simplified Implementation
 *
 * This provides the firedrake_mlir_ext module without complex custom dialects.
 * We use standard MLIR dialects which are sufficient for our needs.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

namespace py = pybind11;

namespace {

// Create an MLIR context with all standard dialects
mlir::MLIRContext* createMLIRContext() {
    auto* context = new mlir::MLIRContext();

    // Register all standard dialects we need
    mlir::registerAllDialects(*context);

    // Load the dialects
    context->loadAllAvailableDialects();

    return context;
}

// Destroy context
void destroyMLIRContext(mlir::MLIRContext* context) {
    delete context;
}

// Create a module
mlir::ModuleOp createModule(mlir::MLIRContext* context) {
    mlir::OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();
    return mlir::ModuleOp::create(loc);
}

} // namespace

PYBIND11_MODULE(firedrake_mlir_ext, m) {
    m.doc() = "Firedrake MLIR Extensions (Simplified)";

    // Context management
    m.def("create_context", []() {
        return py::capsule(createMLIRContext(), [](void* ptr) {
            destroyMLIRContext(static_cast<mlir::MLIRContext*>(ptr));
        });
    });

    m.def("create_module", [](py::capsule context) {
        auto* ctx = static_cast<mlir::MLIRContext*>(context.get_pointer());
        auto module = createModule(ctx);
        return py::capsule(module.getAsOpaquePointer());
    });

    // Information
    m.def("get_registered_dialects", [](py::capsule context) {
        auto* ctx = static_cast<mlir::MLIRContext*>(context.get_pointer());
        std::vector<std::string> dialects;
        for (auto& dialect : ctx->getLoadedDialects()) {
            dialects.push_back(dialect.first.str());
        }
        return dialects;
    });

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("USE_CUSTOM_DIALECTS") = false;  // We use standard MLIR dialects
    m.attr("DIALECT_COUNT") = 17;  // Number of standard dialects we load
}
/*
 * MLIR Test Helpers
 *
 * Additional helper functions for MLIR testing
 */

#include "test_utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace firedrake {
namespace test {

// Helper to apply patterns to a module
bool applyPatternsToModule(ModuleOp module, RewritePatternSet& patterns) {
    GreedyRewriteConfig config;
    return succeeded(applyPatternsGreedily(module, std::move(patterns), config));
}

// Helper to create a simple FEM kernel structure
func::FuncOp createFEMKernelStructure(OpBuilder& builder, ModuleOp module,
                                      const std::string& name,
                                      int numDofs, int numQuadPoints) {
    auto f64Type = builder.getF64Type();

    // Create typical FEM kernel signature
    SmallVector<Type> inputTypes;
    inputTypes.push_back(MemRefType::get({-1, -1}, f64Type));  // Global matrix
    inputTypes.push_back(MemRefType::get({numDofs, numDofs}, f64Type));  // Element matrix
    inputTypes.push_back(MemRefType::get({numDofs, numQuadPoints}, f64Type));  // Basis
    inputTypes.push_back(MemRefType::get({numQuadPoints}, f64Type));  // Weights

    auto func = createTestFunction(builder, module, name, inputTypes, {});

    return func;
}

// Helper to check if module uses advanced MLIR features
bool usesAdvancedMLIRFeatures(ModuleOp module) {
    bool hasVector = containsOp(module, "vector");
    bool hasAffine = containsOp(module, "affine");
    bool hasSparse = containsOp(module, "sparse");
    bool hasPDL = containsOp(module, "pdl");

    return hasVector || hasAffine || hasSparse || hasPDL;
}

// Helper to validate no intermediate layers
bool validateNoIntermediateLayers(ModuleOp module) {
    std::string moduleStr = moduleToString(module);

    // Check for absence of intermediate layer artifacts
    bool noGEM = moduleStr.find("gem") == std::string::npos;
    bool noImpero = moduleStr.find("impero") == std::string::npos;
    bool noLoopy = moduleStr.find("loopy") == std::string::npos;
    bool noCoffee = moduleStr.find("coffee") == std::string::npos;

    // Check for presence of MLIR constructs
    bool hasMLIR = moduleStr.find("func.func") != std::string::npos ||
                   moduleStr.find("module") != std::string::npos;

    return noGEM && noImpero && noLoopy && noCoffee && hasMLIR;
}

// Helper to measure compilation performance
class CompilationTimer {
public:
    CompilationTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~CompilationTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        llvm::outs() << "  " << name_ << " took " << duration.count() << "ms\n";
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Helper to generate test data
std::vector<double> generateTestData(size_t size, double minVal = 0.0, double maxVal = 1.0) {
    std::vector<double> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = minVal + (maxVal - minVal) * (i / double(size));
    }
    return data;
}

} // namespace test
} // namespace firedrake
} // namespace mlir
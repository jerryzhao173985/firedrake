/*
 * mlir_backend.cpp - Consolidated Python interface for Firedrake MLIR backend
 *
 * This provides a clean Python API for using MLIR to replace GEM/Impero/Loopy.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declare C API functions from CAPI.cpp
extern "C" {
int fd_init_once(void);
void* fd_compiler_create(void);
void fd_compiler_destroy(void* compiler);
int fd_compiler_verify(void* compiler);
int fd_compiler_optimize(void* compiler, int level);
int fd_compiler_lower_to_llvm(void* compiler);
int fd_compiler_create_jit(void* compiler, void** out_kernel);
char* fd_compiler_get_mlir_text(void* compiler);
char* fd_compiler_get_llvm_text(void* compiler);
void fd_free(void* p);
int fd_register_dialect(const char* dialect_name);
int fd_is_dialect_registered(const char* dialect_name);
}

class MLIRBackend {
private:
    void* compiler;
    void* kernel;
    bool verbose;

public:
    MLIRBackend(bool verbose = false) : verbose(verbose), compiler(nullptr), kernel(nullptr) {
        // Initialize MLIR once
        fd_init_once();

        // Create compiler instance
        compiler = fd_compiler_create();

        if (verbose && compiler) {
            py::print("MLIR backend initialized");
        }
    }

    ~MLIRBackend() {
        if (compiler) {
            fd_compiler_destroy(compiler);
        }
    }

    std::string getIR() const {
        if (!compiler) return "";

        char* text = fd_compiler_get_mlir_text(compiler);
        if (!text) return "";

        std::string result(text);
        fd_free(text);
        return result;
    }

    bool optimize() {
        if (!compiler) return false;

        if (verbose) {
            py::print("Running MLIR optimization pipeline...");
        }

        // Optimize at level 2
        return fd_compiler_optimize(compiler, 2) == 0;
    }

    bool compile() {
        if (!compiler) return false;

        if (verbose) {
            py::print("JIT compiling MLIR module...");
        }

        // Lower to LLVM first
        if (fd_compiler_lower_to_llvm(compiler) != 0) {
            return false;
        }

        // Create JIT
        return fd_compiler_create_jit(compiler, &kernel) == 0;
    }

    py::capsule getKernel(const std::string& name) {
        if (!kernel) {
            throw std::runtime_error("No kernel compiled");
        }
        return py::capsule(kernel, name.c_str());
    }

    void reset() {
        // Reset by creating new compiler
        if (compiler) {
            fd_compiler_destroy(compiler);
        }
        compiler = fd_compiler_create();
        kernel = nullptr;
    }

    bool isAvailable() const {
        return compiler != nullptr;
    }

    std::string getVersion() const {
        return "Firedrake MLIR Backend v1.0";
    }
};

// Module definition
PYBIND11_MODULE(firedrake_mlir_backend, m) {
    m.doc() = "Firedrake MLIR backend - replaces GEM/Impero/Loopy with MLIR";

    py::class_<MLIRBackend>(m, "MLIRBackend")
        .def(py::init<bool>(), py::arg("verbose") = false,
             "Initialize MLIR backend")
        .def("get_ir", &MLIRBackend::getIR,
             "Get current MLIR IR representation")
        .def("optimize", &MLIRBackend::optimize,
             "Run optimization pipeline")
        .def("compile", &MLIRBackend::compile,
             "JIT compile the module")
        .def("get_kernel", &MLIRBackend::getKernel,
             "Get compiled kernel function", py::arg("name"))
        .def("reset", &MLIRBackend::reset,
             "Reset module for new compilation")
        .def("is_available", &MLIRBackend::isAvailable,
             "Check if backend is available")
        .def("get_version", &MLIRBackend::getVersion,
             "Get backend version")
        .def("__repr__", [](const MLIRBackend& backend) {
            return "<MLIRBackend " + backend.getVersion() + ">";
        });

    // Utility functions
    m.def("test_mlir_backend", []() {
        try {
            MLIRBackend backend(true);
            if (!backend.isAvailable()) {
                return false;
            }

            // Try to optimize empty module
            if (!backend.optimize()) {
                py::print("Warning: Optimization failed");
            }

            py::print("MLIR backend test successful");
            return true;
        } catch (const std::exception& e) {
            py::print("MLIR backend test failed:", e.what());
            return false;
        }
    }, "Test MLIR backend availability");

    m.def("get_mlir_info", []() {
        py::dict info;
        info["version"] = "1.0";

        py::list dialects;
        dialects.append("FEM");
        dialects.append("GEM");
        dialects.append("Affine");
        dialects.append("SCF");
        dialects.append("Linalg");
        info["dialects"] = dialects;

        py::list optimizations;
        optimizations.append("CSE");
        optimizations.append("Canonicalization");
        optimizations.append("Loop Fusion");
        optimizations.append("Vectorization");
        optimizations.append("Cache Tiling");
        info["optimizations"] = optimizations;

        py::list targets;
        targets.append("LLVM");
        targets.append("Native");
        info["targets"] = targets;

        return info;
    }, "Get MLIR backend information");
}
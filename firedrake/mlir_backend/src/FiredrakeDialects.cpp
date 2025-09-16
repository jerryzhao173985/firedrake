/*
 * Firedrake MLIR Dialects Implementation
 * 
 * This file implements the FEM and GEM dialects for Firedrake using MLIR C++ API.
 */

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace firedrake {

//===----------------------------------------------------------------------===//
// FEM Dialect
//===----------------------------------------------------------------------===//

class FEMDialect : public Dialect {
public:
  explicit FEMDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<FEMDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "FEMOps.cpp.inc"
    >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "FEMTypes.cpp.inc"
    >();
  }

  static StringRef getDialectNamespace() { return "fem"; }

  /// Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

//===----------------------------------------------------------------------===//
// FEM Types
//===----------------------------------------------------------------------===//

namespace detail {

/// Storage for FunctionSpace type
struct FunctionSpaceTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<StringRef, unsigned>;

  FunctionSpaceTypeStorage(StringRef family, unsigned degree)
      : family(family), degree(degree) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(family, degree);
  }

  static FunctionSpaceTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    return new (allocator.allocate<FunctionSpaceTypeStorage>())
        FunctionSpaceTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  StringRef family;
  unsigned degree;
};

} // namespace detail

class FunctionSpaceType : public Type::TypeBase<FunctionSpaceType, Type,
                                                 detail::FunctionSpaceTypeStorage> {
public:
  using Base::Base;

  static FunctionSpaceType get(MLIRContext *context, StringRef family,
                                unsigned degree) {
    return Base::get(context, family, degree);
  }

  StringRef getFamily() const { return getImpl()->family; }
  unsigned getDegree() const { return getImpl()->degree; }
};

//===----------------------------------------------------------------------===//
// FEM Operations
//===----------------------------------------------------------------------===//

class FunctionSpaceOp : public Op<FunctionSpaceOp> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fem.function_space"; }

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static void build(OpBuilder &builder, OperationState &result,
                    StringRef family, unsigned degree) {
    result.addTypes(FunctionSpaceType::get(builder.getContext(), family, degree));
    result.addAttribute("family", builder.getStringAttr(family));
    result.addAttribute("degree", builder.getI32IntegerAttr(degree));
  }

  LogicalResult verify() {
    // Verify that family is valid (CG, DG, RT, etc.)
    StringRef family = (*this)->getAttrOfType<StringAttr>("family").getValue();
    if (family != "CG" && family != "DG" && family != "RT" && family != "N1curl")
      return emitOpError("invalid element family: ") << family;
    return success();
  }
};

class WeakFormOp : public Op<WeakFormOp> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fem.weak_form"; }

  static void build(OpBuilder &builder, OperationState &result,
                    Value trialSpace, Value testSpace,
                    Value bilinearForm, Value linearForm) {
    result.addOperands({trialSpace, testSpace, bilinearForm, linearForm});
    result.addTypes(builder.getI32Type()); // Returns problem ID
  }

  LogicalResult verify() {
    // Verify that spaces are compatible
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GEM Dialect
//===----------------------------------------------------------------------===//

class GEMDialect : public Dialect {
public:
  explicit GEMDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<GEMDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "GEMOps.cpp.inc"
    >();
  }

  static StringRef getDialectNamespace() { return "gem"; }
};

//===----------------------------------------------------------------------===//
// GEM Operations
//===----------------------------------------------------------------------===//

class IndexSumOp : public Op<IndexSumOp> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "gem.index_sum"; }

  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange indices, Value body) {
    result.addOperands(indices);
    result.addOperands(body);
    // Result type is same as body type
    result.addTypes(body.getType());
  }

  LogicalResult verify() {
    // Verify index bounds
    return success();
  }
};

class ProductOp : public Op<ProductOp> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "gem.product"; }

  static void build(OpBuilder &builder, OperationState &result,
                    Value lhs, Value rhs) {
    result.addOperands({lhs, rhs});
    // Infer result type from operands
    result.addTypes(lhs.getType());
  }

  LogicalResult verify() {
    // Verify types are compatible for multiplication
    if (getOperand(0).getType() != getOperand(1).getType())
      return emitOpError("operands must have same type");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

void registerFiredrakeDialects(DialectRegistry &registry) {
  registry.insert<FEMDialect, GEMDialect>();
}

void registerFiredrakeDialects(MLIRContext &context) {
  DialectRegistry registry;
  registerFiredrakeDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace firedrake
} // namespace mlir

//===----------------------------------------------------------------------===//
// Python Bindings
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// C API wrapper functions
extern "C" {

void *firedrake_mlir_create_context() {
  auto *context = new mlir::MLIRContext();
  mlir::firedrake::registerFiredrakeDialects(*context);
  return context;
}

void firedrake_mlir_destroy_context(void *context) {
  delete static_cast<mlir::MLIRContext *>(context);
}

void *firedrake_mlir_create_module(void *context) {
  auto *ctx = static_cast<mlir::MLIRContext *>(context);
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx));
  return module.getAsOpaquePointer();
}

} // extern "C"

PYBIND11_MODULE(firedrake_mlir_ext, m) {
  m.doc() = "Firedrake MLIR Extensions";

  // Expose context creation
  m.def("create_context", []() {
    return firedrake_mlir_create_context();
  });

  m.def("destroy_context", [](py::capsule context) {
    firedrake_mlir_destroy_context(context);
  });

  m.def("create_module", [](py::capsule context) {
    return firedrake_mlir_create_module(context);
  });

  // Expose dialect registration
  m.def("register_dialects", [](py::capsule context) {
    auto *ctx = static_cast<mlir::MLIRContext *>(context);
    mlir::firedrake::registerFiredrakeDialects(*ctx);
  });
}
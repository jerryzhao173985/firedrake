//===- FEMOps.td - FEM dialect operations ----------------*- tablegen -*-===//
//
// This file defines the operations for the FEM dialect using MLIR's TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef FEM_OPS
#define FEM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// FEM Dialect definition
//===----------------------------------------------------------------------===//

def FEM_Dialect : Dialect {
  let name = "fem";
  let summary = "Finite Element Method dialect";
  let description = [{
    The FEM dialect provides high-level operations for finite element
    computations, including function spaces, weak forms, and integrals.
  }];
  let cppNamespace = "::mlir::firedrake::fem";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// FEM Types
//===----------------------------------------------------------------------===//

class FEM_Type<string name, string typeMnemonic> : TypeDef<FEM_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def FEM_FunctionSpaceType : FEM_Type<"FunctionSpace", "function_space"> {
  let summary = "Finite element function space";
  let description = [{
    Represents a finite element function space defined on a mesh with a
    specific element family (CG, DG, RT, etc.) and polynomial degree.
  }];

  let parameters = (ins
    StringRefParameter<"element family">:$family,
    "unsigned":$degree
  );

  let assemblyFormat = "`<` $family `,` $degree `>`";
}

def FEM_FormType : FEM_Type<"Form", "form"> {
  let summary = "Differential form";
  let description = [{
    Represents a differential form (0-form, 1-form, or 2-form).
  }];

  let parameters = (ins "unsigned":$rank);
  let assemblyFormat = "`<` $rank `>`";
}

//===----------------------------------------------------------------------===//
// FEM Operations
//===----------------------------------------------------------------------===//

class FEM_Op<string mnemonic, list<Trait> traits = []> :
    Op<FEM_Dialect, mnemonic, traits>;

def FEM_FunctionSpaceOp : FEM_Op<"function_space", [Pure]> {
  let summary = "Create a function space";
  let description = [{
    Creates a finite element function space on a mesh with the specified
    element family and polynomial degree.

    Example:
    ```mlir
    %V = fem.function_space "CG" 1 : !fem.function_space<"CG", 1>
    ```
  }];

  let arguments = (ins
    StrAttr:$family,
    I32Attr:$degree
  );

  let results = (outs FEM_FunctionSpaceType:$space);

  let assemblyFormat = [{
    $family $degree attr-dict `:` type($space)
  }];

  let hasVerifier = 1;
}

def FEM_TrialFunctionOp : FEM_Op<"trial_function", [Pure]> {
  let summary = "Create a trial function";
  let description = [{
    Creates a trial function in the specified function space.
  }];

  let arguments = (ins FEM_FunctionSpaceType:$space);
  let results = (outs AnyTensor:$function);

  let assemblyFormat = [{
    $space attr-dict `:` type($space) `->` type($function)
  }];
}

def FEM_TestFunctionOp : FEM_Op<"test_function", [Pure]> {
  let summary = "Create a test function";
  let description = [{
    Creates a test function in the specified function space.
  }];

  let arguments = (ins FEM_FunctionSpaceType:$space);
  let results = (outs AnyTensor:$function);

  let assemblyFormat = [{
    $space attr-dict `:` type($space) `->` type($function)
  }];
}

def FEM_GradientOp : FEM_Op<"gradient", [Pure]> {
  let summary = "Compute gradient of a function";
  let description = [{
    Computes the gradient of a function.

    Example:
    ```mlir
    %grad_u = fem.gradient %u : tensor<?xf64> -> tensor<?x2xf64>
    ```
  }];

  let arguments = (ins AnyTensor:$function);
  let results = (outs AnyTensor:$gradient);

  let assemblyFormat = [{
    $function attr-dict `:` type($function) `->` type($gradient)
  }];
}

def FEM_InnerProductOp : FEM_Op<"inner", [Pure]> {
  let summary = "Inner product of two functions";
  let description = [{
    Computes the inner product of two functions or vectors.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def FEM_IntegralOp : FEM_Op<"integral", [Pure]> {
  let summary = "Define an integral";
  let description = [{
    Defines an integral over a domain (cell, exterior facet, or interior facet).

    Example:
    ```mlir
    %a = fem.integral "cell" {
      %inner = fem.inner %grad_u, %grad_v : ...
      fem.yield %inner
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$domain,
    OptionalAttr<I32Attr>:$subdomain_id
  );

  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $domain (`subdomain` `=` $subdomain_id^)? attr-dict-with-keyword 
    $body `:` type($result)
  }];
}

// Simplified weak form operation
def FEM_WeakFormOp : FEM_Op<"weak_form", []> {
  let summary = "Define a weak form";
  let description = [{
    Defines a weak formulation of a PDE with forms.
  }];

  let arguments = (ins
    FEM_FunctionSpaceType:$test_space,
    I32:$form_id
  );

  let results = (outs I32:$problem_id);

  let assemblyFormat = [{
    `test` `=` $test_space
    `form` `=` $form_id
    attr-dict `:` type($test_space) `->` type($problem_id)
  }];
}

def FEM_YieldOp : FEM_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield a value from an integral";
  let description = [{
    Yields a value from within an integral region.
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

#endif // FEM_OPS
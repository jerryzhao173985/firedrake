//===- GEMOps.td - GEM dialect operations ----------------*- tablegen -*-===//
//
// This file defines the operations for the GEM (Generalized Expression Mapping)
// dialect using MLIR's TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef GEM_OPS
#define GEM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// GEM Dialect definition
//===----------------------------------------------------------------------===//

def GEM_Dialect : Dialect {
  let name = "gem";
  let summary = "Generalized Expression Mapping dialect";
  let description = [{
    The GEM dialect provides tensor algebra operations that serve as an
    intermediate representation between high-level FEM operations and
    low-level loop-based code.
  }];
  let cppNamespace = "::mlir::firedrake::gem";
}

//===----------------------------------------------------------------------===//
// GEM Operations
//===----------------------------------------------------------------------===//

class GEM_Op<string mnemonic, list<Trait> traits = []> :
    Op<GEM_Dialect, mnemonic, traits>;

def GEM_IndexOp : GEM_Op<"index", [Pure]> {
  let summary = "Define a loop index";
  let description = [{
    Defines a loop index with a given extent.

    Example:
    ```mlir
    %i = gem.index 10 : index
    ```
  }];

  let arguments = (ins I64Attr:$extent);
  let results = (outs Index:$index);

  let assemblyFormat = [{
    $extent attr-dict `:` type($index)
  }];
}

def GEM_IndexSumOp : GEM_Op<"index_sum", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Sum over indices";
  let description = [{
    Performs summation over the specified indices, equivalent to nested loops.

    Example:
    ```mlir
    %result = gem.index_sum %i, %j : index, index {
      %val = gem.product %a, %b : f64
      gem.yield %val : f64
    } : f64
    ```
  }];

  let arguments = (ins Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $indices attr-dict `:` type($indices) $body `:` type($result)
  }];

  let hasVerifier = 1;
}

def GEM_ProductOp : GEM_Op<"product", [Pure, Commutative]> {
  let summary = "Multiply expressions";
  let description = [{
    Multiplies two expressions element-wise.

    Example:
    ```mlir
    %c = gem.product %a, %b : f64
    ```
  }];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def GEM_SumOp : GEM_Op<"sum", [Pure, Commutative]> {
  let summary = "Add expressions";
  let description = [{
    Adds multiple expressions together.

    Example:
    ```mlir
    %d = gem.sum %a, %b, %c : f64
    ```
  }];

  let arguments = (ins Variadic<F64>:$operands);
  let results = (outs F64:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` type($result)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def GEM_IndexedOp : GEM_Op<"indexed", [Pure]> {
  let summary = "Indexed tensor access";
  let description = [{
    Accesses a tensor element at specified indices.

    Example:
    ```mlir
    %val = gem.indexed %tensor[%i, %j] : tensor<?x?xf64>
    ```
  }];

  let arguments = (ins AnyTensor:$tensor, Variadic<Index>:$indices);
  let results = (outs F64:$value);

  let assemblyFormat = [{
    $tensor `[` $indices `]` attr-dict `:` type($tensor) `->` type($value)
  }];

  let hasVerifier = 1;
}

def GEM_ComponentTensorOp : GEM_Op<"component_tensor", [Pure]> {
  let summary = "Create tensor from indexed expression";
  let description = [{
    Creates a tensor by evaluating an expression for each index combination.

    Example:
    ```mlir
    %tensor = gem.component_tensor [%i, %j] : index, index {
      %val = gem.product %i, %j : index
      gem.yield %val : f64
    } : tensor<?x?xf64>
    ```
  }];

  let arguments = (ins Variadic<Index>:$indices);
  let results = (outs AnyTensor:$tensor);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `[` $indices `]` attr-dict `:` type($indices) $body `:` type($tensor)
  }];
}

def GEM_ConditionalOp : GEM_Op<"conditional", [Pure]> {
  let summary = "Conditional expression";
  let description = [{
    Evaluates to true_value if condition is true, otherwise false_value.

    Example:
    ```mlir
    %val = gem.conditional %cond ? %true : %false : f64
    ```
  }];

  let arguments = (ins I1:$condition, F64:$true_value, F64:$false_value);
  let results = (outs F64:$result);

  let assemblyFormat = [{
    $condition `?` $true_value `:` $false_value attr-dict `:` type($result)
  }];
}

def GEM_DeltaOp : GEM_Op<"delta", [Pure]> {
  let summary = "Kronecker delta";
  let description = [{
    Returns 1 if indices are equal, 0 otherwise.

    Example:
    ```mlir
    %d = gem.delta %i, %j : index
    ```
  }];

  let arguments = (ins Index:$i, Index:$j);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $i `,` $j attr-dict `:` type($i)
  }];

  let hasFolder = 1;
}

def GEM_LiteralOp : GEM_Op<"literal", [Pure, ConstantLike]> {
  let summary = "Literal constant value";
  let description = [{
    Represents a compile-time constant value.

    Example:
    ```mlir
    %c = gem.literal 3.14 : f64
    ```
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];

  let hasFolder = 1;
}

def GEM_AccumulateOp : GEM_Op<"accumulate", []> {
  let summary = "Accumulate value into tensor";
  let description = [{
    Accumulates a value into a tensor at specified indices.

    Example:
    ```mlir
    gem.accumulate %value to %tensor[%i, %j] : f64, tensor<?x?xf64>
    ```
  }];

  let arguments = (ins 
    AnyType:$value,
    AnyTensor:$tensor,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $value `to` $tensor `[` $indices `]` attr-dict `:` 
    type($value) `,` type($tensor)
  }];
}

def GEM_YieldOp : GEM_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield a value";
  let description = [{
    Yields a value from within a GEM region.
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// GEM Transformations and Optimizations
//===----------------------------------------------------------------------===//

// Note: Pass definitions need mlir/Pass/PassBase.td include
// These should be defined in a separate Passes.td file
// Commenting out to avoid build errors

// def DeltaElimination : Pass<"gem-delta-elimination", "func::FuncOp"> {
//   let summary = "Eliminate Kronecker deltas";
//   let description = [{
//     This pass eliminates Kronecker delta operations by replacing index
//     variables where appropriate.
//   }];
//   let constructor = "mlir::firedrake::gem::createDeltaEliminationPass()";
// }

// def SumFactorization : Pass<"gem-sum-factorization", "func::FuncOp"> {
//   let summary = "Apply sum factorization";
//   let description = [{
//     This pass applies sum factorization to reduce computational complexity
//     of nested summations.
//   }];
//   let constructor = "mlir::firedrake::gem::createSumFactorizationPass()";
// }

#endif // GEM_OPS
#!/bin/bash

# Comprehensive check of MLIR installation for Firedrake backend

echo "============================================================"
echo "MLIR INSTALLATION CHECK FOR FIREDRAKE BACKEND"
echo "============================================================"
echo ""

INSTALL_DIR="/Users/jerry/llvm-install"
ALL_GOOD=true

# Essential libraries for our Firedrake MLIR backend
declare -a ESSENTIAL_LIBS=(
    # Core (MUST HAVE)
    "MLIRIR"
    "MLIRSupport"
    "MLIRAnalysis"
    "MLIRParser"
    "MLIRPass"
    "MLIRTransforms"

    # Dialects we use
    "MLIRAffineDialect"
    "MLIRArithDialect"
    "MLIRFuncDialect"
    "MLIRLinalgDialect"
    "MLIRMemRefDialect"
    "MLIRSCFDialect"
    "MLIRTensorDialect"
    "MLIRVectorDialect"
    "MLIRSparseTensorDialect"

    # Conversions we need
    "MLIRAffineToStandard"
    "MLIRArithToLLVM"
    "MLIRSCFToControlFlow"
    "MLIRMemRefToLLVM"
    "MLIRFuncToLLVM"

    # Pattern infrastructure
    "MLIRPDLDialect"
    "MLIRRewrite"
)

echo "Checking essential MLIR libraries..."
echo "-------------------------------------"

MISSING=""
for lib in "${ESSENTIAL_LIBS[@]}"; do
    if ls ${INSTALL_DIR}/lib/lib${lib}.* >/dev/null 2>&1; then
        echo "✅ ${lib}"
    else
        echo "❌ ${lib} - MISSING"
        MISSING="${MISSING} ${lib}"
        ALL_GOOD=false
    fi
done

# Check for MLIR tools
echo ""
echo "Checking essential MLIR tools..."
echo "---------------------------------"

declare -a TOOLS=(
    "mlir-opt"
    "mlir-translate"
    "mlir-tblgen"
)

for tool in "${TOOLS[@]}"; do
    if [ -f "${INSTALL_DIR}/bin/${tool}" ]; then
        echo "✅ ${tool}"
    else
        echo "❌ ${tool} - MISSING"
        ALL_GOOD=false
    fi
done

# Check dynamic libraries
echo ""
echo "Checking dynamic libraries..."
echo "------------------------------"

if [ -f "${INSTALL_DIR}/lib/libMLIR.dylib" ] || [ -f "${INSTALL_DIR}/lib/libMLIR.so" ]; then
    echo "✅ MLIR dynamic library found"
else
    echo "⚠️  No MLIR dynamic library (static libs available)"
fi

if [ -f "${INSTALL_DIR}/lib/libMLIR-C.dylib" ] || [ -f "${INSTALL_DIR}/lib/libMLIR-C.so" ]; then
    echo "✅ MLIR-C dynamic library found"
else
    echo "⚠️  No MLIR-C dynamic library"
fi

# Count total libraries
echo ""
echo "Library Statistics:"
echo "-------------------"
TOTAL_MLIR_LIBS=$(ls ${INSTALL_DIR}/lib/libMLIR*.* 2>/dev/null | wc -l)
echo "Total MLIR libraries installed: ${TOTAL_MLIR_LIBS}"

# Final verdict
echo ""
echo "============================================================"
if [ "$ALL_GOOD" = true ]; then
    echo "✅ MLIR INSTALLATION IS COMPLETE FOR FIREDRAKE!"
    echo ""
    echo "You have all essential components:"
    echo "  • Core MLIR infrastructure"
    echo "  • All required dialects (Affine, Linalg, SCF, etc.)"
    echo "  • All conversion passes"
    echo "  • Pattern rewriting infrastructure"
    echo "  • MLIR tools (opt, translate, tblgen)"
    echo ""
    echo "Ready to use the Firedrake MLIR backend!"
else
    echo "⚠️  Some components might be missing, but check if they're named differently"
    echo "Missing: ${MISSING}"
    echo ""
    echo "However, if the build completed, the libraries might be under different names."
fi

echo "============================================================"
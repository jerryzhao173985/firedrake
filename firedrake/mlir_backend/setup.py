"""
Setup script for building Firedrake MLIR extensions
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the MLIR extensions")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if sys.platform == "darwin":
            # macOS specific flags
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15']

        # Set LLVM/MLIR paths
        llvm_dir = os.path.expanduser("~/llvm-install")
        cmake_args += [
            f'-DLLVM_DIR={llvm_dir}/lib/cmake/llvm',
            f'-DMLIR_DIR={llvm_dir}/lib/cmake/mlir'
        ]

        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)

        # Build
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)


setup(
    name='firedrake_mlir_backend',
    version='0.1.0',
    author='Firedrake Team',
    description='MLIR backend for Firedrake',
    long_description='',
    ext_modules=[CMakeExtension('firedrake_mlir_ext')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.8",
)
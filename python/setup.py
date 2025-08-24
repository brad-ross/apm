from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "apm._core",
        [
            "src/apm/_core_enhanced.cpp",
            "../core/src/apm_core.cpp",
            "../core/src/linear_algebra_utils.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "../core/src",
        ],
        libraries=["armadillo", "boost_graph", "arrow", "blas", "lapack"],
        library_dirs=[os.environ.get("CONDA_PREFIX", "") + "/lib"],
        cxx_std=17,
        define_macros=[
            ("ARMA_DONT_USE_WRAPPER", "1"),
            ("ARMA_NO_DEBUG", "1"),
            ("ARMA_USE_LAPACK", "1"),
            ("ARMA_USE_BLAS", "1"),
        ],
    ),
]

setup(
    name="apm",
    version="0.1.0",
    author="Brad Ross, Lihua Lei, Apoorva Lal",
    author_email="rossb@nber.org",
    description="Python bindings for the Aggregated Projection Matrix method",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["apm"],
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)

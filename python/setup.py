import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core" / "src"

libraries = ["blas", "lapack"]
extra_link_args = ["-static-libstdc++", "-static-libgcc"]
if sys.platform == "darwin":
    libraries = []
    extra_link_args = ["-framework", "Accelerate"]

ext_modules = [
    Pybind11Extension(
        "apm._core",
        [
            "src/apm/_core.cpp",
            "../core/src/apm_core.cpp",
            "../core/src/bootstrap.cpp",
            "../core/src/linear_algebra_utils.cpp",
            "../core/src/online_accumulators.cpp",
            "../core/src/utils.cpp",
            "../core/src/factor_model_estimators/FactorModelEstimator.cpp",
            "../core/src/factor_model_estimators/pc_estimators.cpp",
        ],
        include_dirs=[str(CORE)],
        libraries=libraries,
        extra_link_args=extra_link_args,
        cxx_std=17,
        define_macros=[
            ("ARMA_DONT_USE_WRAPPER", "1"),
            ("ARMA_NO_DEBUG", "1"),
            ("ARMA_USE_LAPACK", "1"),
            ("ARMA_USE_BLAS", "1"),
        ],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["apm"],
    package_dir={"": "src"},
)

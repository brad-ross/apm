cmake_minimum_required(VERSION 3.16)
project(apm_python)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Support conda/mamba environments
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
endif()

# Find dependencies
find_package(pybind11 REQUIRED)

# Configure Armadillo as header-only
set(ARMA_DONT_USE_WRAPPER 1)
find_package(Armadillo REQUIRED)

# Find Boost (for Graph Library)
find_package(Boost REQUIRED COMPONENTS graph)

# Find Arrow
find_package(Arrow REQUIRED)

# Add the core library sources directly (since we're building as part of Python module)
set(APM_CORE_SOURCES
    ../core/src/apm_core.cpp
    ../core/src/linear_algebra_utils.cpp
)

# Create the Python module
pybind11_add_module(_core src/apm/_core.cpp ${APM_CORE_SOURCES})

# Include directories
target_include_directories(_core PRIVATE ../core/src)

# Link libraries
target_link_libraries(_core PRIVATE
    ${ARMADILLO_LIBRARIES}
    Boost::graph
    Arrow::arrow_shared
)

# Compiler flags for Armadillo
target_compile_definitions(_core PRIVATE
    ARMA_DONT_USE_WRAPPER=1
    ARMA_NO_DEBUG=1
    ARMA_USE_LAPACK=1
    ARMA_USE_BLAS=1
)

# Explicitly link Arrow compute libraries if needed
find_library(ARROW_ACERO_LIB NAMES arrow_acero libarrow_acero)
find_library(ARROW_COMPUTE_LIB NAMES arrow_compute libarrow_compute)
if(ARROW_ACERO_LIB)
    target_link_libraries(_core PRIVATE ${ARROW_ACERO_LIB})
endif()
if(ARROW_COMPUTE_LIB)
    target_link_libraries(_core PRIVATE ${ARROW_COMPUTE_LIB})
endif()

# Version info
target_compile_definitions(_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")
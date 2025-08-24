cmake_minimum_required(VERSION 3.16)
project(apm_core VERSION 0.1.0 LANGUAGES CXX)

# Modern CMake policies
cmake_policy(SET CMP0167 OLD)  # Keep using FindBoost for now

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Support conda/mamba environments
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
endif()

# Configure Armadillo as header-only
set(ARMA_DONT_USE_WRAPPER 1)
find_package(Armadillo REQUIRED)

# Find BLAS and LAPACK (required by Armadillo)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Find Boost (for Graph Library) - conda-friendly
find_package(Boost REQUIRED COMPONENTS graph)

# Find Arrow - conda-friendly
find_package(Arrow REQUIRED)

# Arrow found successfully

# Create the core library
add_library(apm src/apm_core.cpp src/linear_algebra_utils.cpp)
target_include_directories(apm PUBLIC src)

# Link libraries using modern CMake targets
target_link_libraries(apm
    PUBLIC
        ${ARMADILLO_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        Boost::graph
        Arrow::arrow_shared
)

# Explicitly link Acero and Compute libraries (needed for Homebrew Arrow)
find_library(ARROW_ACERO_LIB NAMES arrow_acero libarrow_acero)
find_library(ARROW_COMPUTE_LIB NAMES arrow_compute libarrow_compute)
if(ARROW_ACERO_LIB)
    target_link_libraries(apm PUBLIC ${ARROW_ACERO_LIB})
endif()
if(ARROW_COMPUTE_LIB)
    target_link_libraries(apm PUBLIC ${ARROW_COMPUTE_LIB})
endif()

# On macOS, link against the Accelerate framework for BLAS/LAPACK
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(apm PUBLIC ${ACCELERATE_LIBRARY})
    else()
        message(WARNING "Accelerate framework not found on Apple system.")
    endif()
endif()

# Add compiler flags for Armadillo
target_compile_definitions(apm PUBLIC
    ARMA_DONT_USE_WRAPPER=1
    ARMA_NO_DEBUG=1
    ARMA_USE_LAPACK=1
    ARMA_USE_BLAS=1
)

# Enable testing
enable_testing()

# Google Test
find_package(GTest QUIET)
if(GTest_FOUND)
    add_subdirectory(tests)
else()
    message(WARNING "Google Test not found. Tests will not be built.")
endif()

# Installation
install(TARGETS apm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES src/apm_core.h src/linear_algebra_utils.h
    DESTINATION include/apm
) 
cmake_minimum_required(VERSION 3.16)
project(apm_core VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure Armadillo as header-only
set(ARMA_DONT_USE_WRAPPER 1)
find_package(Armadillo REQUIRED)

# Find Boost (for Graph Library)
find_package(Boost REQUIRED COMPONENTS graph)

# Find Arrow using system installation (Homebrew)
find_package(Arrow REQUIRED)

# Find oneTBB for parallelization (optional; Homebrew installs TBBConfig.cmake)
find_package(TBB CONFIG QUIET COMPONENTS tbb)

# Arrow found successfully

# Create the core library
add_library(apm
    src/apm_core.cpp
    src/est_outcome_means.cpp
    src/linear_algebra_utils.cpp
    src/bootstrap.cpp
    # factor_model_parameter_structs is header-only now
    src/factor_model_estimators/FactorModelEstimator.cpp
    src/factor_model_estimators/pc_estimators.cpp
    src/OutcomeMeanSuffStatEstimator.cpp
    src/nuisance_param_estimators.cpp
    src/est_cohort_specific_params.cpp
    src/utils.cpp
)
target_include_directories(apm PUBLIC src)

# Link libraries using modern CMake targets
target_link_libraries(apm
    PUBLIC
        ${ARMADILLO_LIBRARIES}
        Boost::graph
        Arrow::arrow_shared
)

if(TARGET TBB::tbb)
    message(STATUS "oneTBB found; building with parallel cohort loop")
    target_link_libraries(apm PUBLIC TBB::tbb)
    target_compile_definitions(apm PUBLIC APM_HAS_TBB=1)
else()
    message(STATUS "oneTBB NOT found; building with serial cohort loop")
endif()

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

install(FILES src/apm_core.h src/est_outcome_means.h src/bootstrap.h src/factor_model_estimators/FactorModelEstimator.h src/factor_model_estimators/pc_estimators.h src/cohort_specific_param_structs.h src/OutcomeMeanSuffStatEstimator.h src/nuisance_param_estimators.h src/est_cohort_specific_params.h src/utils.h
    DESTINATION include/apm
) 
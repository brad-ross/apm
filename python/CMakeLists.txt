cmake_minimum_required(VERSION 3.16)
project(apm_python VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)

set(APM_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../core/src")
set(APM_CORE_SOURCES
    "${APM_CORE_DIR}/apm_core.cpp"
    "${APM_CORE_DIR}/bootstrap.cpp"
    "${APM_CORE_DIR}/linear_algebra_utils.cpp"
    "${APM_CORE_DIR}/online_accumulators.cpp"
    "${APM_CORE_DIR}/utils.cpp"
    "${APM_CORE_DIR}/factor_model_estimators/FactorModelEstimator.cpp"
    "${APM_CORE_DIR}/factor_model_estimators/pc_estimators.cpp"
)

pybind11_add_module(_core src/apm/_core.cpp ${APM_CORE_SOURCES})
target_include_directories(_core PRIVATE "${APM_CORE_DIR}")
target_link_libraries(_core PRIVATE
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    Boost::graph
)
target_compile_definitions(_core PRIVATE
    ARMA_DONT_USE_WRAPPER=1
    ARMA_NO_DEBUG=1
    ARMA_USE_LAPACK=1
    ARMA_USE_BLAS=1
)

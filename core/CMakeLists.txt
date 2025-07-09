cmake_minimum_required(VERSION 3.16)
project(apm_core VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Configure Armadillo as header-only
set(ARMA_DONT_USE_WRAPPER 1)
find_package(Armadillo REQUIRED)

# Find Boost (for Graph Library)
find_package(Boost REQUIRED COMPONENTS graph)

# Create the core library
add_library(apm_core STATIC
    src/apm.cpp
    src/linear_algebra_utils.cpp
)

target_include_directories(apm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(apm_core PUBLIC
    ${ARMADILLO_LIBRARIES}
    Boost::boost
)

# On macOS, link against the Accelerate framework for BLAS/LAPACK
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(apm_core PUBLIC ${ACCELERATE_LIBRARY})
    else()
        message(WARNING "Accelerate framework not found on Apple system.")
    endif()
endif()

# Add compiler flags for Armadillo
target_compile_definitions(apm_core PUBLIC
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
install(TARGETS apm_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES src/apm.h
    DESTINATION include/apm
) 
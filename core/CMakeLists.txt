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
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find Boost (for Graph Library)
find_package(Boost REQUIRED COMPONENTS graph)
include_directories(${Boost_INCLUDE_DIRS})

# Create the core library
add_library(apm src/apm.cpp src/linear_algebra_utils.cpp)
target_include_directories(apm PUBLIC src)

# Link libraries
target_link_libraries(apm
    PUBLIC
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
)

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

install(FILES src/apm.h
    DESTINATION include/apm
) 
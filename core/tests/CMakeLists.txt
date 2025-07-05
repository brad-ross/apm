# CMakeLists.txt for C++ Tests
cmake_minimum_required(VERSION 3.16)

# Find Google Test
find_package(GTest REQUIRED)

# Create test executable
add_executable(test_apm_core
    test_apm.cpp
)

# Link with the core library and Google Test
target_link_libraries(test_apm_core
    apm_core
    GTest::gtest
    GTest::gtest_main
    ${ARMADILLO_LIBRARIES}
)

# Include directories
target_include_directories(test_apm_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${ARMADILLO_INCLUDE_DIRS}
)

# Add test
add_test(
    NAME CoreTests
    COMMAND test_apm_core
)

# Enable test discovery
include(GoogleTest)
gtest_discover_tests(test_apm_core) 
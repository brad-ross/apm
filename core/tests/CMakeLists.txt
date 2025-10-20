# CMakeLists.txt for C++ Tests
cmake_minimum_required(VERSION 3.16)

# Find Google Test
find_package(GTest REQUIRED)

# Create test executable
add_executable(test_apm
    test_helpers.cpp
    test_est_outcome_means.cpp
    test_agg_cohort_specific_factor_model_params.cpp
    test_apm_core.cpp
    test_summarize_identification.cpp
    test_linear_algebra_utils.cpp
    test_bootstrap.cpp
    test_pc_estimators.cpp
    test_twfe_estimator.cpp
    test_outcome_mean_suff_stat_estimators.cpp
    test_est_cohort_specific_params.cpp
    test_outcome_imputation.cpp
    test_est_target_param_components.cpp
    test_cluster_outcomes.cpp
)

# Link with the core library and Google Test
target_link_libraries(test_apm
    PRIVATE
    apm
    GTest::gtest
)

# Include directories
target_include_directories(test_apm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Enable test-only APIs
target_compile_definitions(test_apm PRIVATE APM_TESTS=1)

# Add test
add_test(
    NAME CoreTests
    COMMAND test_apm
)

# Enable test discovery
include(GoogleTest)
gtest_discover_tests(test_apm) 
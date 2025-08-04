#include <gtest/gtest.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <vector>
#include <set>
#include <stdexcept>
#include <memory>
#include <string>
#include <filesystem>
#include "process_panel.h"

namespace {

/**
 * @brief Create test dataset with staircase cohort pattern
 * 
 * Creates a dataset with the following structure:
 * - 3 cohorts following staircase pattern: {0,1,2}, {1,2,3}, {2,3,4}
 * - 2 units per cohort (6 total units)
 * - Units are named: unit_A, unit_B (cohort 0), unit_C, unit_D (cohort 1), unit_E, unit_F (cohort 2)
 * - Outcomes are named: outcome_0, outcome_1, outcome_2, outcome_3, outcome_4
 * - Each unit-outcome combination gets a unique value
 */
std::shared_ptr<arrow::Table> create_staircase_test_dataset() {
    // Builders for each column
    auto unit_builder = std::make_shared<arrow::StringBuilder>();
    auto outcome_builder = std::make_shared<arrow::StringBuilder>();
    auto value_builder = std::make_shared<arrow::DoubleBuilder>();
    
    // Unit names for easier debugging
    std::vector<std::string> unit_names = {"unit_A", "unit_B", "unit_C", "unit_D", "unit_E", "unit_F"};
    
    // Cohort 0: units A,B observe outcomes 0,1,2
    std::vector<int> cohort_0_outcomes = {0, 1, 2};
    for (int unit_idx = 0; unit_idx < 2; ++unit_idx) {
        for (int outcome : cohort_0_outcomes) {
            EXPECT_TRUE(unit_builder->Append(unit_names[unit_idx]).ok());
            EXPECT_TRUE(outcome_builder->Append("outcome_" + std::to_string(outcome)).ok());
            // Use predictable values: unit_index * 100 + outcome * 10 + 1
            EXPECT_TRUE(value_builder->Append(unit_idx * 100 + outcome * 10 + 1).ok());
        }
    }
    
    // Cohort 1: units C,D observe outcomes 1,2,3
    std::vector<int> cohort_1_outcomes = {1, 2, 3};
    for (int unit_idx = 2; unit_idx < 4; ++unit_idx) {
        for (int outcome : cohort_1_outcomes) {
            EXPECT_TRUE(unit_builder->Append(unit_names[unit_idx]).ok());
            EXPECT_TRUE(outcome_builder->Append("outcome_" + std::to_string(outcome)).ok());
            EXPECT_TRUE(value_builder->Append(unit_idx * 100 + outcome * 10 + 1).ok());
        }
    }
    
    // Cohort 2: units E,F observe outcomes 2,3,4
    std::vector<int> cohort_2_outcomes = {2, 3, 4};
    for (int unit_idx = 4; unit_idx < 6; ++unit_idx) {
        for (int outcome : cohort_2_outcomes) {
            EXPECT_TRUE(unit_builder->Append(unit_names[unit_idx]).ok());
            EXPECT_TRUE(outcome_builder->Append("outcome_" + std::to_string(outcome)).ok());
            EXPECT_TRUE(value_builder->Append(unit_idx * 100 + outcome * 10 + 1).ok());
        }
    }
    
    // Finish arrays
    std::shared_ptr<arrow::Array> unit_array, outcome_array, value_array;
    EXPECT_TRUE(unit_builder->Finish(&unit_array).ok());
    EXPECT_TRUE(outcome_builder->Finish(&outcome_array).ok());
    EXPECT_TRUE(value_builder->Finish(&value_array).ok());
    
    // Create schema
    auto schema = arrow::schema({
        arrow::field("unit_id", arrow::utf8()),
        arrow::field("outcome", arrow::utf8()),
        arrow::field("value", arrow::float64())
    });
    
    // Create table
    return arrow::Table::Make(schema, {unit_array, outcome_array, value_array});
}

/**
 * @brief Verify that the cohort assignment matches expected staircase pattern
 */
void verify_staircase_cohorts(
    const std::vector<std::set<size_t>>& observed_outcome_indices,
    std::shared_ptr<arrow::Array> outcome_names) {
    
    // Should have exactly 3 cohorts
    ASSERT_EQ(observed_outcome_indices.size(), 3);
    
    // Convert outcome names to strings for easier verification
    auto string_array = std::static_pointer_cast<arrow::StringArray>(outcome_names);
    std::vector<std::string> outcome_strings;
    for (int64_t i = 0; i < string_array->length(); ++i) {
        outcome_strings.push_back(string_array->GetString(i));
    }
    
    // Verify outcomes are in correct order: outcome_0, outcome_1, outcome_2, outcome_3, outcome_4
    ASSERT_EQ(outcome_strings.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(outcome_strings[i], "outcome_" + std::to_string(i));
    }
    
    // Verify cohort patterns
    // Cohort 0: {0, 1, 2}
    std::set<size_t> expected_cohort_0 = {0, 1, 2};
    EXPECT_EQ(observed_outcome_indices[0], expected_cohort_0);
    
    // Cohort 1: {1, 2, 3}
    std::set<size_t> expected_cohort_1 = {1, 2, 3};
    EXPECT_EQ(observed_outcome_indices[1], expected_cohort_1);
    
    // Cohort 2: {2, 3, 4}
    std::set<size_t> expected_cohort_2 = {2, 3, 4};
    EXPECT_EQ(observed_outcome_indices[2], expected_cohort_2);
}

/**
 * @brief Verify that cohort table has correct unit assignments and sort order
 */
void verify_cohort_table_structure(std::shared_ptr<arrow::Table> cohort_table) {
    ASSERT_EQ(cohort_table->num_rows(), 6);  // 6 units total
    ASSERT_EQ(cohort_table->num_columns(), 2);  // unit_id, cohort_id
    
    // Get columns
    auto unit_column = std::static_pointer_cast<arrow::StringArray>(cohort_table->column(0)->chunk(0));
    auto cohort_column = std::static_pointer_cast<arrow::Int32Array>(cohort_table->column(1)->chunk(0));
    
    // Verify table is sorted by cohort_id, then by unit_id
    for (int64_t i = 0; i < cohort_table->num_rows() - 1; ++i) {
        int32_t current_cohort = cohort_column->Value(i);
        int32_t next_cohort = cohort_column->Value(i + 1);
        
        if (current_cohort == next_cohort) {
            // Same cohort: units should be in ascending order
            std::string current_unit = unit_column->GetString(i);
            std::string next_unit = unit_column->GetString(i + 1);
            EXPECT_LT(current_unit, next_unit) 
                << "Units within cohort should be sorted: " << current_unit << " vs " << next_unit;
        } else {
            // Different cohort: cohort_id should be ascending
            EXPECT_LT(current_cohort, next_cohort)
                << "Cohorts should be in ascending order: " << current_cohort << " vs " << next_cohort;
        }
    }
    
    // Verify specific assignments
    // Cohort 0: unit_A (row 0), unit_B (row 1)
    EXPECT_EQ(cohort_column->Value(0), 0);
    EXPECT_EQ(unit_column->GetString(0), "unit_A");
    EXPECT_EQ(cohort_column->Value(1), 0);
    EXPECT_EQ(unit_column->GetString(1), "unit_B");
    
    // Cohort 1: unit_C (row 2), unit_D (row 3)
    EXPECT_EQ(cohort_column->Value(2), 1);
    EXPECT_EQ(unit_column->GetString(2), "unit_C");
    EXPECT_EQ(cohort_column->Value(3), 1);
    EXPECT_EQ(unit_column->GetString(3), "unit_D");
    
    // Cohort 2: unit_E (row 4), unit_F (row 5)
    EXPECT_EQ(cohort_column->Value(4), 2);
    EXPECT_EQ(unit_column->GetString(4), "unit_E");
    EXPECT_EQ(cohort_column->Value(5), 2);
    EXPECT_EQ(unit_column->GetString(5), "unit_F");
}

} // anonymous namespace

//==============================================================================
// construct_panel_cohorts Tests
//==============================================================================

TEST(ProcessPanelTest, ConstructPanelCohorts_StaircasePattern) {
    auto dataset = create_staircase_test_dataset();
    
    auto [cohort_table, observed_outcome_indices, outcome_names] = 
        apm::construct_panel_cohorts(dataset, "unit_id", "outcome");
    
    // Verify the cohort structure matches staircase pattern
    verify_staircase_cohorts(observed_outcome_indices, outcome_names);
    
    // Verify cohort table structure and sorting
    verify_cohort_table_structure(cohort_table);
}

TEST(ProcessPanelTest, ConstructPanelCohorts_EmptyDataset) {
    // Create empty dataset
    auto schema = arrow::schema({
        arrow::field("unit_id", arrow::utf8()),
        arrow::field("outcome", arrow::utf8())
    });
    std::vector<std::shared_ptr<arrow::Array>> empty_columns;
    // Use the overload that takes schema, columns vector, and optional num_rows
    auto empty_table = arrow::Table::Make(schema, empty_columns, 0);
    
    EXPECT_THROW(
        apm::construct_panel_cohorts(empty_table, "unit_id", "outcome"),
        std::runtime_error
    );
}

TEST(ProcessPanelTest, ConstructPanelCohorts_MissingColumn) {
    auto dataset = create_staircase_test_dataset();
    
    EXPECT_THROW(
        apm::construct_panel_cohorts(dataset, "nonexistent_unit", "outcome"),
        std::runtime_error
    );
    
    EXPECT_THROW(
        apm::construct_panel_cohorts(dataset, "unit_id", "nonexistent_outcome"),
        std::runtime_error
    );
}

//==============================================================================
// UnbalancedPanel Tests
//==============================================================================

TEST(ProcessPanelTest, UnbalancedPanel_InMemoryStorage) {
    auto dataset = create_staircase_test_dataset();
    
    apm::UnbalancedPanel panel(
        dataset, 
        "unit_id", 
        "outcome", 
        "value",
        apm::PanelStorageStrategy::COPY_IN_MEMORY
    );
    
    // Test basic properties
    EXPECT_EQ(panel.get_num_cohorts(), 3);
    EXPECT_EQ(panel.get_num_units(), 6);
    EXPECT_EQ(panel.get_storage_strategy(), apm::PanelStorageStrategy::COPY_IN_MEMORY);
    
    // Verify cohort structure
    verify_staircase_cohorts(panel.get_observed_outcome_indices(), panel.get_outcome_names());
    verify_cohort_table_structure(panel.get_cohort_table());
    
    // Test that sorted dataset has correct basic structure
    auto sorted_dataset = panel.get_sorted_dataset();
    EXPECT_EQ(sorted_dataset->num_rows(), 18);  // 6 units * 3 outcomes each
    EXPECT_EQ(sorted_dataset->num_columns(), 4);  // unit_id, outcome, value, cohort_id
}

TEST(ProcessPanelTest, UnbalancedPanel_DiskStorage) {
    auto dataset = create_staircase_test_dataset();
    
    // Create temporary file path
    std::string temp_path = std::filesystem::temp_directory_path() / "test_panel_data.arrow";
    
    {
        apm::UnbalancedPanel panel(
            dataset, 
            "unit_id", 
            "outcome", 
            "value",
            apm::PanelStorageStrategy::COPY_ON_DISK,
            temp_path
        );
        
        // Test basic properties
        EXPECT_EQ(panel.get_num_cohorts(), 3);
        EXPECT_EQ(panel.get_num_units(), 6);
        EXPECT_EQ(panel.get_storage_strategy(), apm::PanelStorageStrategy::COPY_ON_DISK);
        
        // Verify cohort structure
        verify_staircase_cohorts(panel.get_observed_outcome_indices(), panel.get_outcome_names());
        verify_cohort_table_structure(panel.get_cohort_table());
        
        // Test sorted dataset from disk
        auto sorted_dataset = panel.get_sorted_dataset();
        EXPECT_EQ(sorted_dataset->num_rows(), 18);
        EXPECT_EQ(sorted_dataset->num_columns(), 4);
    }
    
    // Clean up temporary file
    if (std::filesystem::exists(temp_path)) {
        std::filesystem::remove(temp_path);
    }
}

TEST(ProcessPanelTest, UnbalancedPanel_DiskStorageMissingPath) {
    auto dataset = create_staircase_test_dataset();
    
    EXPECT_THROW(
        apm::UnbalancedPanel panel(
            dataset, 
            "unit_id", 
            "outcome", 
            "value",
            apm::PanelStorageStrategy::COPY_ON_DISK
            // Missing disk_path parameter
        ),
        std::runtime_error
    );
}

TEST(ProcessPanelTest, SingleUnitSingleOutcome) {
    // Create minimal dataset
    auto unit_builder = std::make_shared<arrow::StringBuilder>();
    auto outcome_builder = std::make_shared<arrow::StringBuilder>();
    auto value_builder = std::make_shared<arrow::DoubleBuilder>();
    
    EXPECT_TRUE(unit_builder->Append("unit_A").ok());
    EXPECT_TRUE(outcome_builder->Append("outcome_0").ok());
    EXPECT_TRUE(value_builder->Append(42.0).ok());
    
    std::shared_ptr<arrow::Array> unit_array, outcome_array, value_array;
    EXPECT_TRUE(unit_builder->Finish(&unit_array).ok());
    EXPECT_TRUE(outcome_builder->Finish(&outcome_array).ok());
    EXPECT_TRUE(value_builder->Finish(&value_array).ok());
    
    auto schema = arrow::schema({
        arrow::field("unit_id", arrow::utf8()),
        arrow::field("outcome", arrow::utf8()),
        arrow::field("value", arrow::float64())
    });
    
    auto dataset = arrow::Table::Make(schema, {unit_array, outcome_array, value_array});
    
    apm::UnbalancedPanel panel(dataset, "unit_id", "outcome", "value");
    
    EXPECT_EQ(panel.get_num_cohorts(), 1);
    EXPECT_EQ(panel.get_num_units(), 1);
    
    auto observed_outcomes = panel.get_observed_outcome_indices();
    EXPECT_EQ(observed_outcomes.size(), 1);
    EXPECT_EQ(observed_outcomes[0], std::set<size_t>{0});
}

// Note: main() function is provided by test_apm_core.cpp since all tests are compiled together
#ifndef PROCESS_PANEL_H
#define PROCESS_PANEL_H

#include <arrow/api.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace apm {

/**
 * Storage strategy for sorted dataset in UnbalancedPanel.
 */
enum class PanelStorageStrategy {
    COPY_IN_MEMORY,   // Store sorted copy in memory (fast access, 2x memory)
    COPY_ON_DISK      // Store sorted copy on disk (handles large datasets)
};

/**
 * Processes unbalanced panel data to identify unit cohorts based on observed outcomes.
 * 
 * This function takes an Arrow dataset containing unbalanced panel data and identifies
 * which cohort each unit belongs to based on the set of outcomes observed for that unit.
 * A cohort is defined as a unique combination of observed outcomes across all units.
 * 
 * The function works with any data type for unit and outcome identifiers, internally
 * mapping them to integer indices for processing.
 * 
 * @param dataset Input Arrow dataset containing panel data
 * @param unit_col Name of the column that uniquely identifies units
 * @param outcome_col Name of the column that uniquely identifies outcomes
 * @return A tuple containing:
 *         - A new Arrow dataset with one row per unit, containing:
 *           * The unit identifier (original type)
 *           * An integer cohort identifier (0-based) uniquely identifying the cohort
 *         - A vector of sets representing observed_outcome_indices for each cohort
 *         - An Arrow array containing unique outcome names/identifiers
 * 
 * @throws std::runtime_error if the dataset is empty, columns don't exist, or other errors
 */
std::tuple<std::shared_ptr<arrow::Table>, std::vector<std::set<size_t>>, std::shared_ptr<arrow::Array>> construct_panel_cohorts(
    std::shared_ptr<arrow::Table> dataset,
    const std::string& unit_col,
    const std::string& outcome_col
);

/**
 * Class for managing unbalanced panel data with cohort information.
 * 
 * This class provides a convenient interface for working with unbalanced panel data
 * that has been processed to identify unit cohorts. It stores the original dataset,
 * cohort information, and provides access to the processed data.
 */
class UnbalancedPanel {
public:
    /**
     * Constructor that processes the dataset to identify cohorts.
     * 
     * @param dataset Input Arrow dataset containing panel data
     * @param unit_col Name of the column that uniquely identifies units
     * @param outcome_col Name of the column that uniquely identifies outcomes
     * @param outcome_value_col Name of the column containing outcome values
     * @param storage_strategy How to store the sorted dataset (view, memory copy, or disk copy)
     * @param disk_path Path for disk storage (required only if storage_strategy is COPY_ON_DISK)
     * @throws std::runtime_error if the dataset is empty, columns don't exist, or other errors
     */
    UnbalancedPanel(
        std::shared_ptr<arrow::Table> dataset,
        const std::string& unit_col,
        const std::string& outcome_col,
        const std::string& outcome_value_col,
        PanelStorageStrategy storage_strategy = PanelStorageStrategy::COPY_IN_MEMORY,
        std::optional<std::string> disk_path = std::nullopt
    );

    /**
     * Get the original dataset.
     * 
     * @return Pointer to the original Arrow dataset
     */
    std::shared_ptr<arrow::Table> get_original_dataset() const { return original_dataset_; }

    /**
     * Get the cohort table with unit-cohort assignments.
     * 
     * @return Arrow table with unit identifiers and cohort assignments
     */
    std::shared_ptr<arrow::Table> get_cohort_table() const { return cohort_table_; }

    /**
     * Get the observed outcome indices for each cohort.
     * 
     * @return Vector of sets representing observed outcome indices for each cohort
     */
    const std::vector<std::set<size_t>>& get_observed_outcome_indices() const { 
        return observed_outcome_indices_; 
    }

    /**
     * Get the unique outcome names/identifiers.
     * 
     * @return Arrow array containing unique outcome names in index order
     */
    std::shared_ptr<arrow::Array> get_outcome_names() const { return outcome_names_; }

    /**
     * Get the dataset in cohort-sorted order.
     * This method provides a unified interface regardless of storage strategy.
     * 
     * @return Arrow table with data sorted by cohort
     */
    std::shared_ptr<arrow::Table> get_sorted_dataset() const;

    /**
     * Get the storage strategy used by this panel.
     * 
     * @return The storage strategy
     */
    PanelStorageStrategy get_storage_strategy() const { return storage_strategy_; }



    /**
     * Get the number of cohorts.
     * 
     * @return Number of unique cohorts
     */
    size_t get_num_cohorts() const { return observed_outcome_indices_.size(); }

    /**
     * Get the number of units.
     * 
     * @return Number of units in the dataset
     */
    size_t get_num_units() const { return cohort_table_->num_rows(); }

private:
    std::shared_ptr<arrow::Table> original_dataset_;
    std::shared_ptr<arrow::Table> cohort_table_;
    std::vector<std::set<size_t>> observed_outcome_indices_;
    std::shared_ptr<arrow::Array> outcome_names_;
    PanelStorageStrategy storage_strategy_;
    
    // Storage strategy specific members
    std::shared_ptr<arrow::Table> sorted_dataset_;    // For COPY_IN_MEMORY
    std::optional<std::string> disk_path_;            // For COPY_ON_DISK
};

} // namespace apm

#endif // PROCESS_PANEL_H 
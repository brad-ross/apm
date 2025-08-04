#include "process_panel.h"
#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/compute/api.h>
#include <arrow/compute/initialize.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <stdexcept>

namespace apm {

// Helper functions in order of usage
namespace {

namespace ac = arrow::acero;
namespace cp = arrow::compute;

// Helper function to ensure Arrow compute functions are initialized
void ensure_arrow_compute_initialized() {
    static bool initialized = false;
    if (!initialized) {
        // Initialize Arrow compute functions
        auto init_status = arrow::compute::Initialize();
        if (!init_status.ok()) {
            throw std::runtime_error("Failed to initialize Arrow compute: " + init_status.ToString());
        }
        
        // Verify required functions are available
        auto* registry = arrow::compute::GetFunctionRegistry();
        std::vector<std::string> required_functions = {"sort_indices", "add", "is_null"};
        
        for (const auto& func_name : required_functions) {
            auto func = registry->GetFunction(func_name);
            if (!func.ok()) {
                throw std::runtime_error("Required Arrow compute function '" + func_name + "' is not available. "
                    "Arrow compute functions may not be properly initialized.");
            }
        }
        
        initialized = true;
    }
}

void internal_status_check(const arrow::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error("Arrow operation failed: " + status.ToString());
    }
}

// Helper: Sort a table using Acero
std::shared_ptr<arrow::Table> sort_table(
    std::shared_ptr<arrow::Table> table,
    const cp::SortOptions& sort_options) {

    ensure_arrow_compute_initialized();
    auto plan_result = ac::ExecPlan::Make(*cp::threaded_exec_context());
    internal_status_check(plan_result.status());
    auto plan = plan_result.ValueOrDie();

    auto source_options = ac::TableSourceNodeOptions{table};
    auto source_result = ac::MakeExecNode("table_source", plan.get(), {}, source_options);
    internal_status_check(source_result.status());
    auto source_node = source_result.ValueOrDie();

    arrow::AsyncGenerator<std::optional<cp::ExecBatch>> sink_gen;
    auto sink_result = ac::MakeExecNode("order_by_sink", plan.get(), {source_node},
        ac::OrderBySinkNodeOptions{sort_options, &sink_gen});
    internal_status_check(sink_result.status());

    internal_status_check(plan->Validate());
    plan->StartProducing();

    std::shared_ptr<arrow::RecordBatchReader> reader = 
        ac::MakeGeneratorReader(source_node->output_schema(), std::move(sink_gen), arrow::default_memory_pool());

    auto table_result = reader->ToTable();
    internal_status_check(table_result.status());
    auto sorted_table = table_result.ValueOrDie();

    plan->StopProducing();
    auto finished_future = plan->finished();
    internal_status_check(finished_future.status());

    return sorted_table;
}

// Helper 1: Collect observed outcomes for a unit as indices
std::set<size_t> collect_unit_outcome_indices(
    const std::shared_ptr<arrow::ChunkedArray>& sorted_outcome_array,
    const std::shared_ptr<arrow::Array>& unique_outcome_array,
    size_t start_row, size_t end_row) {

    std::set<size_t> unit_outcome_indices;

    for (size_t i = start_row; i < end_row; ++i) {
        auto outcome_scalar_res = sorted_outcome_array->GetScalar(i);
        internal_status_check(outcome_scalar_res.status());
        auto outcome_scalar = outcome_scalar_res.ValueUnsafe();

        if (outcome_scalar->is_valid) {
            // Find the index of this outcome in the unique outcomes array
            for (size_t j = 0; j < unique_outcome_array->length(); ++j) {
                auto unique_scalar = unique_outcome_array->GetScalar(j).ValueOrDie();
                if (outcome_scalar->Equals(*unique_scalar)) {
                    unit_outcome_indices.insert(j);
                    break;
                }
            }
        }
    }

    return unit_outcome_indices;
}

// Helper 2: Find the index of a cohort in observed_outcome_indices
size_t find_cohort_index(const std::set<size_t>& unit_outcome_indices,
                        const std::vector<std::set<size_t>>& observed_outcome_indices) {
    for (size_t i = 0; i < observed_outcome_indices.size(); ++i) {
        if (unit_outcome_indices == observed_outcome_indices[i]) {
            return i;
        }
    }
    return observed_outcome_indices.size(); // New cohort
}

// Helper 3: Create subset of original dataset with essential columns (including outcome_value)
std::shared_ptr<arrow::Table> create_essential_subset(
    std::shared_ptr<arrow::Table> original_dataset,
    const std::string& unit_col,
    const std::string& outcome_col,
    const std::string& outcome_value_col) {

    auto unit_col_idx = original_dataset->schema()->GetFieldIndex(unit_col);
    auto outcome_col_idx = original_dataset->schema()->GetFieldIndex(outcome_col);
    auto outcome_value_col_idx = original_dataset->schema()->GetFieldIndex(outcome_value_col);

    auto subset_schema = arrow::schema({
        original_dataset->schema()->field(unit_col_idx),
        original_dataset->schema()->field(outcome_col_idx),
        original_dataset->schema()->field(outcome_value_col_idx)
    });

    return arrow::Table::Make(subset_schema, {
        original_dataset->column(unit_col_idx),
        original_dataset->column(outcome_col_idx),
        original_dataset->column(outcome_value_col_idx)
    });
}

// Helper 4: Join dataset with cohort table and sort by cohort using Acero
std::shared_ptr<arrow::Table> join_and_sort_by_cohort(
    std::shared_ptr<arrow::Table> dataset,
    std::shared_ptr<arrow::Table> cohort_table,
    const std::string& unit_col,
    const std::string& outcome_col,
    const std::string& outcome_value_col) {

    // Create the execution plan
    auto plan_result = ac::ExecPlan::Make(*cp::threaded_exec_context());
    internal_status_check(plan_result.status());
    auto plan = plan_result.ValueOrDie();

    // Create source nodes using TableSourceNodeOptions
    auto dataset_source_options = ac::TableSourceNodeOptions{dataset};
    auto dataset_source_result = ac::MakeExecNode("table_source", plan.get(), {}, dataset_source_options);
    internal_status_check(dataset_source_result.status());
    auto dataset_source = dataset_source_result.ValueOrDie();

    auto cohort_source_options = ac::TableSourceNodeOptions{cohort_table};
    auto cohort_source_result = ac::MakeExecNode("table_source", plan.get(), {}, cohort_source_options);
    internal_status_check(cohort_source_result.status());
    auto cohort_source = cohort_source_result.ValueOrDie();

    // Create hash join node
    auto join_options = ac::HashJoinNodeOptions{
        ac::JoinType::INNER,
        /*left_keys=*/{unit_col},
        /*right_keys=*/{unit_col}
    };
    auto join_result = ac::MakeExecNode("hashjoin", plan.get(), {dataset_source, cohort_source}, join_options);
    internal_status_check(join_result.status());
    auto join_node = join_result.ValueOrDie();



    // Create projection node to remove duplicate unit_col from right side
    // Join output: [left: unit_id, outcome, value] + [right: unit_id, cohort_id]
    // We want:     [left: unit_id, outcome, value] + [right: cohort_id] (drop duplicate unit_id)
    
    // Use field references by position to avoid naming conflicts
    std::vector<cp::Expression> projection_exprs = {
        cp::field_ref(0),  // unit_id from left (dataset)
        cp::field_ref(1),  // outcome from left (dataset)  
        cp::field_ref(2),  // value from left (dataset)
        cp::field_ref(4)   // cohort_id from right (skip position 3 which is duplicate unit_id)
    };
    
    // Provide explicit names for the projected columns using the actual column names
    std::vector<std::string> projection_names = {unit_col, outcome_col, outcome_value_col, "cohort_id"};
    
    auto project_options = ac::ProjectNodeOptions{projection_exprs, projection_names};
    auto project_result = ac::MakeExecNode("project", plan.get(), {join_node}, project_options);
    internal_status_check(project_result.status());
    auto project_node = project_result.ValueOrDie();



    // Create order by sink node
    arrow::AsyncGenerator<std::optional<cp::ExecBatch>> sink_gen;
    auto sink_result = ac::MakeExecNode("order_by_sink", plan.get(), {project_node},
        ac::OrderBySinkNodeOptions{
            cp::SortOptions{{
                cp::SortKey{"cohort_id", cp::SortOrder::Ascending},
                cp::SortKey{unit_col, cp::SortOrder::Ascending}  // This should work after projection
            }},
            &sink_gen
        });
    internal_status_check(sink_result.status());

    // Execute the plan
    internal_status_check(plan->Validate());
    plan->StartProducing();

    // Collect results using MakeGeneratorReader
    std::shared_ptr<arrow::RecordBatchReader> reader = 
        ac::MakeGeneratorReader(project_node->output_schema(), std::move(sink_gen), arrow::default_memory_pool());

    auto table_result = reader->ToTable();
    internal_status_check(table_result.status());
    auto final_table = table_result.ValueOrDie();

    // Clean up
    plan->StopProducing();
    auto finished_future = plan->finished();
    internal_status_check(finished_future.status());

    return final_table;
}
} // anonymous namespace

std::tuple<std::shared_ptr<arrow::Table>, std::vector<std::set<size_t>>, std::shared_ptr<arrow::Array>> construct_panel_cohorts(
    std::shared_ptr<arrow::Table> dataset,
    const std::string& unit_col,
    const std::string& outcome_col
) {
    ensure_arrow_compute_initialized();
    
    // Validate input
    if (!dataset || dataset->num_rows() == 0) {
        throw std::runtime_error("Dataset is empty or null");
    }

    // Check if required columns exist
    auto unit_col_idx = dataset->schema()->GetFieldIndex(unit_col);
    auto outcome_col_idx = dataset->schema()->GetFieldIndex(outcome_col);

    if (unit_col_idx == -1) {
        throw std::runtime_error("Unit column '" + unit_col + "' not found in dataset");
    }
    if (outcome_col_idx == -1) {
        throw std::runtime_error("Outcome column '" + outcome_col + "' not found in dataset");
    }

    // Extract unit and outcome columns
    auto unit_array = dataset->column(unit_col_idx);
    auto outcome_array = dataset->column(outcome_col_idx);

    // Step 1: Create sorted dataset by unit column, then by outcome column within each unit
    cp::SortOptions sort_options;
    sort_options.sort_keys = {
        cp::SortKey(unit_col, cp::SortOrder::Ascending),
        cp::SortKey(outcome_col, cp::SortOrder::Ascending)
    };
    auto sorted_dataset = sort_table(dataset, sort_options);

    auto sorted_unit_array = sorted_dataset->column(unit_col_idx);
    auto sorted_outcome_array = sorted_dataset->column(outcome_col_idx);

    // Step 2: Get unique outcomes array
    auto unique_outcome_array = arrow::compute::Unique(sorted_outcome_array).ValueOrDie();

    // Step 3: Process units and build cohorts
    std::vector<std::set<size_t>> observed_outcome_indices;

    // Builders for the unit_cohort_table
    auto unit_builder = arrow::MakeBuilder(sorted_unit_array->type()).ValueOrDie();
    auto cohort_builder = std::make_shared<arrow::Int32Builder>();

    size_t current_unit_start = 0;
    size_t num_rows = sorted_dataset->num_rows();

    while (current_unit_start < num_rows) {
        // Find the end of the current unit's data
        size_t current_unit_end = current_unit_start;
        auto current_unit_scalar = sorted_unit_array->GetScalar(current_unit_start).ValueOrDie();

        while (current_unit_end < num_rows) {
            auto next_unit_scalar = sorted_unit_array->GetScalar(current_unit_end).ValueOrDie();
            if (!current_unit_scalar->Equals(*next_unit_scalar)) {
                break;
            }
            current_unit_end++;
        }

        // Collect observed outcomes for this unit as indices
        std::set<size_t> unit_outcome_indices = collect_unit_outcome_indices(
            sorted_outcome_array, unique_outcome_array, current_unit_start, current_unit_end);

        // Find cohort index by comparing with existing cohorts
        size_t cohort_id = find_cohort_index(unit_outcome_indices, observed_outcome_indices);

        // If it's a new cohort, add it to the list
        if (cohort_id == observed_outcome_indices.size()) {
            observed_outcome_indices.push_back(unit_outcome_indices);
        }

        // Add unit to result
        internal_status_check(unit_builder->AppendScalar(*current_unit_scalar));
        internal_status_check(cohort_builder->Append(static_cast<int32_t>(cohort_id)));

        // Move to next unit
        current_unit_start = current_unit_end;
    }

    // Build the unit_cohort_table arrays
    std::shared_ptr<arrow::Array> unit_result_array;
    std::shared_ptr<arrow::Array> cohort_result_array;
    internal_status_check(unit_builder->Finish(&unit_result_array));
    internal_status_check(cohort_builder->Finish(&cohort_result_array));

    // Create unit_cohort_table schema
    auto unit_cohort_schema = arrow::schema({
        arrow::field(unit_col, unit_array->type()),
        arrow::field("cohort_id", arrow::int32())
    });

    // Create unit_cohort_table
    auto unit_cohort_table = arrow::Table::Make(unit_cohort_schema, {unit_result_array, cohort_result_array});

    // Step 4: Sort unit_cohort_table by cohort_id, then by unit_col
    arrow::compute::SortOptions sort_options_final;
    sort_options_final.sort_keys = {
        arrow::compute::SortKey("cohort_id", arrow::compute::SortOrder::Ascending),
        arrow::compute::SortKey(unit_col, arrow::compute::SortOrder::Ascending)
    };
    auto final_unit_cohort_table = sort_table(unit_cohort_table, sort_options_final);

    return {final_unit_cohort_table, observed_outcome_indices, unique_outcome_array};
}

// UnbalancedPanel implementation
UnbalancedPanel::UnbalancedPanel(
    std::shared_ptr<arrow::Table> dataset,
    const std::string& unit_col,
    const std::string& outcome_col,
    const std::string& outcome_value_col,
    PanelStorageStrategy storage_strategy,
    std::optional<std::string> disk_path
) : original_dataset_(dataset), storage_strategy_(storage_strategy), disk_path_(disk_path) {

    // Process the dataset to get cohort information
    auto result = construct_panel_cohorts(dataset, unit_col, outcome_col);
    cohort_table_ = std::get<0>(result);
    observed_outcome_indices_ = std::get<1>(result);
    outcome_names_ = std::get<2>(result);

    // Handle different storage strategies with optimized logic
    switch (storage_strategy_) {
        case PanelStorageStrategy::COPY_IN_MEMORY: {
            // Create sorted copy with only essential columns using helper functions
            auto subset_dataset = create_essential_subset(original_dataset_, unit_col, outcome_col, outcome_value_col);
            sorted_dataset_ = join_and_sort_by_cohort(subset_dataset, cohort_table_, unit_col, outcome_col, outcome_value_col);
            break;
        }

        case PanelStorageStrategy::COPY_ON_DISK: {
            // Validate that disk_path is provided
            if (!disk_path_.has_value() || disk_path_->empty()) {
                throw std::runtime_error("disk_path is required for COPY_ON_DISK storage strategy");
            }

            // Create sorted copy with only essential columns and write to disk
            auto subset_dataset = create_essential_subset(original_dataset_, unit_col, outcome_col, outcome_value_col);
            auto sorted_dataset = join_and_sort_by_cohort(subset_dataset, cohort_table_, unit_col, outcome_col, outcome_value_col);

            // Write to disk using Arrow IPC format
            auto output_stream = arrow::io::FileOutputStream::Open(*disk_path_).ValueOrDie();
            auto writer = arrow::ipc::MakeFileWriter(output_stream, sorted_dataset->schema()).ValueOrDie();
            internal_status_check(writer->WriteTable(*sorted_dataset));
            internal_status_check(writer->Close());
            internal_status_check(output_stream->Close());
            break;
        }
    }
}

std::shared_ptr<arrow::Table> UnbalancedPanel::get_sorted_dataset() const {
    switch (storage_strategy_) {
        case PanelStorageStrategy::COPY_IN_MEMORY: {
            // Return pre-computed sorted dataset
            return sorted_dataset_;
        }

        case PanelStorageStrategy::COPY_ON_DISK: {
            // Read from disk
            if (!disk_path_.has_value()) {
                throw std::runtime_error("No disk path available for COPY_ON_DISK storage strategy");
            }
            auto input_stream = arrow::io::ReadableFile::Open(*disk_path_).ValueOrDie();
            auto reader = arrow::ipc::RecordBatchFileReader::Open(input_stream).ValueOrDie();
            return reader->ToTable().ValueOrDie();
        }
    }

    throw std::runtime_error("Unknown storage strategy");
}



} // namespace apm

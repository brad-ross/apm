using Random, DataFrames, ProgressMeter, StatsBase, Distributions, LinearAlgebra

include("./panel_struct.jl")
include("./bootstrap.jl")

function comp_obs_cohort_outcome_mean_lookup(panel::UnbalancedPanel)
    panel_cohorts = get_cohorts(panel)
    panel_outcomes = get_outcomes(panel)
    cohort_outcome_mean_lookup = create_dict_with_types_like(first(panel_cohorts), 
        create_dict_with_types_like(first(panel_outcomes), 0.0))
    cohort_outcome_mean_lookup_lock = ReentrantLock()
    progress_bar = Progress(Base.length(panel_cohorts))
    Threads.@threads for cohort in panel_cohorts
        lock(cohort_outcome_mean_lookup_lock) do
            cohort_outcome_mean_lookup[cohort] = create_dict_with_types_like(first(panel_outcomes), 0.0)
        end
        
        panel_cohort = UnbalancedPanelCohort(panel, cohort)
        obs_cohort_outcome_means = combine(groupby(panel_cohort.cohort_data_df, panel_cohort.time_col), 
            panel_cohort.value_col => mean => :outcome_mean)
        lock(cohort_outcome_mean_lookup_lock) do
            for row in eachrow(obs_cohort_outcome_means)
                cohort_outcome_mean_lookup[cohort][row[panel_cohort.time_col]] = row.outcome_mean
            end
        end
        next!(progress_bar)
    end
    finish!(progress_bar)

    cohort_outcome_mean_lookup
end

function comp_cohort_outcome_mean_ests(panel::UnbalancedPanel, estimators::Dict{String, Function})
    empty_time_id_arr = empty_id_arr(panel.time_idx_map)
    cohort_outcome_mean_ests = DataFrame(est_name = String[], cohort = Set(empty_time_id_arr), outcome_mean = Float64[])
    cohort_outcome_mean_ests[!, panel.time_col] = empty_time_id_arr
    
    for (est_name, est_fn) in estimators
        est_results = est_fn(panel)
        est_results.est_name .= est_name
        append!(cohort_outcome_mean_ests, est_results)
    end

    cohort_outcome_mean_ests
end

function comp_ests(panel::UnbalancedPanel, estimators::Dict{String, Function})
    ests = nothing
    for (est_name, est_fn) in estimators
        est_results = est_fn(panel)
        est_results.est_name .= est_name

        if isnothing(ests)
            ests = est_results
        else
            append!(ests, est_results)
        end
    end
    ests
end

function get_sim_ests(panel_resampler::DataGeneratingProcess, estimators::Dict{String, Function}, B = 100)
    get_sim_ests(panel_resampler, (panel) -> comp_ests(panel, estimators), B)
end
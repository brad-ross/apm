using Random, Parquet, DataFrames, ProgressMeter, StatsBase, Distributions, LinearAlgebra, CSV

include("code/vwh_data_helpers.jl")
include("code/panel_struct.jl")
include("code/simulation_utils.jl")
include("code/estimators.jl")

const RAW_DATA_PATH = joinpath(ENV["DATA_PATH"], "raw_data")
const CLEAN_DATA_PATH = joinpath(ENV["DATA_PATH"], "clean_data")

const OUTPUT_PATH = ENV["OUTPUT_PATH"]

Random.seed!(1337)

function read_panel_data(start_year::Int, year_cluster_size::Int, num_clusters::Int)
    earnings = read_raw_earnings_data()
    firms = read_raw_firm_data()
    joined_data = get_joined_and_filtered_match_data(earnings, firms, start_year, 2001, VENETO_PROVINCES)
    joined_data.year_cluster = get_year_clusters(joined_data.year, year_cluster_size)
    filter_joined_data_to_always_present_workers_and_firms!(joined_data)

    num_firms = nrow(unique(select(joined_data, :firm)))
    println("Number of Unique Firms: $(num_firms)")

    num_firms_per_province = combine(groupby(
        unique(select(joined_data, :firm, :province)), 
        :province), nrow => :num_firms)
    println("Number of Unique Firms per Province: $(num_firms_per_province)")

    num_workers = nrow(unique(select(joined_data, :worker)))
    println("Number of Unique Workers: $(num_workers)")
    
    firm_clustering_path = joinpath(CLEAN_DATA_PATH, "clustering_connectivity_exploration_provinces", 
        "start_year=$(start_year)", "year_cluster_size=$(year_cluster_size)", "num_clusters=$(num_clusters)")
    clustered_firms = DataFrame(read_parquet(firm_clustering_path))
    
    joined_data_with_clusters = @chain joined_data begin
        leftjoin(select(clustered_firms, Not(:num_workers)), on = [:firm, :province])
    end

    avg_weekly_earn_by_type = get_avg_weekly_earn_by_group(joined_data_with_clusters, [:year_cluster, :province, :firm_type])
    subset!(avg_weekly_earn_by_type, :avg_weekly_earnings => (e -> e .> 0))
    transform!(
        avg_weekly_earn_by_type, [:year_cluster, :province, :firm_type] => 
        ((y, p, t) -> string.(t) .* "_" .* p .* "_" .* string.(y)) => :outcome,
        :avg_weekly_earnings => (e -> log.(e)) => :log_avg_weekly_earnings
    )
end

function get_cohort_spectra(panel::UnbalancedPanel)
    cohort_spectra = DataFrame(cohort_str = String[], cohort_size = Int[], 
        spectrum_idx = Int[], eigenvalue = Float64[])
    cohorts = get_cohorts(panel)
    @showprogress for cohort in cohorts
        panel_cohort = UnbalancedPanelCohort(panel, cohort)
        outcome_mat = get_cohort_outcome_mat(panel_cohort)
        outcome_covar_mat = outcome_mat' * outcome_mat / Base.size(outcome_mat, 1)
        eigenvalues = sort(eigvals(Symmetric(outcome_covar_mat)), rev=true)

        cohort_str = join(sort(collect(cohort)), "; ")
        for s in 1:Base.size(outcome_mat, 2)
            push!(cohort_spectra, (cohort_str = cohort_str, cohort_size = Base.size(outcome_mat, 1), 
                spectrum_idx = s, eigenvalue = eigenvalues[s]))
        end
    end

    sort!(cohort_spectra, :cohort_size, rev = true)
end

function est_for_sim(panel::UnbalancedPanel, estimator::Function)
    raw_ests = estimator(panel)
    raw_ests.parameter .= "target_cohort_outcome_mean"
    select!(raw_ests, :parameter, :outcome_mean => :estimate)
end

function main()
    panel_data = read_panel_data(1998, 2, 3)
    max_r = 2

    orig_panel = UnbalancedPanel(panel_data, :worker, :outcome, :log_avg_weekly_earnings, max_r, 75)
    panel = constr_panel_from_identified_outcomes(orig_panel, max_r)
    orig_panel = nothing
    write_parquet(joinpath(CLEAN_DATA_PATH, "final_panel.parquet"), panel.data_df)
    
    # print identified outcomes
    show(sort(collect(get_identified_outcomes(panel, max_r))))
    # print data frame of cohort sizes
    show(get_cohort_sizes(panel))

    cohort_spectra = get_cohort_spectra(panel)
    CSV.write(joinpath(OUTPUT_PATH, "cohort_sizes_and_spectra.csv"), cohort_spectra)

    cohort_outcome_dists = get_cohort_overlap_graph_outcome_distances(panel, 1)
    select!(cohort_outcome_dists, 
        :cohort => (c -> join.(sort.(collect.(c)), "; ")) => :cohort_str, 
        :outcome, :distance => :distance_rank_1)
    CSV.write(joinpath(OUTPUT_PATH, "cohort_outcome_dists.csv"), cohort_outcome_dists)
    
    obs_cohort_outcome_mean_lookup = comp_obs_cohort_outcome_mean_lookup(panel)

    sim_output_path = joinpath(OUTPUT_PATH, "leave_one_out_sims_provinces")
    mkpath(sim_output_path)
    
    GC.gc()

    cohort_count = 0
    all_boot_sim_ests = DataFrame(
        :b => Int64[],
        :cohort_str => String[],
        :cohort_size => Int64[],
        panel.time_col => empty_id_arr(panel.time_idx_map),
        :est_name => String[],
        :boot_outcome_mean => Float64[],
        :true_outcome_mean => Float64[]
    )
    for target_cohort_with_size in eachrow(get_cohort_sizes(panel))
        target_cohort = target_cohort_with_size.cohort
        target_cohort_str = join(sort(collect(target_cohort)), "; ")
        if cohort_count < 15 && Base.length(target_cohort) > max_r
            target_cohort_size = target_cohort_with_size.size
            println("===== cohort: $target_cohort; n = $target_cohort_size =====")
            outcomes_simulated_for_cohort = 0
            for target_outcome in target_cohort
                cohort_outcome_file_path = joinpath(sim_output_path, "boot_sim_ests-$(target_cohort_str)-$(target_outcome).parquet")
                if !isfile(cohort_outcome_file_path)
                    leave_out_panel = constr_panel_with_left_out_cohort_outcome(panel, target_cohort, target_outcome)
                    if target_outcome ∈ get_identified_outcomes(leave_out_panel, max_r)
                        println("=== outcome: $target_outcome ===")
                        estimators = Dict(
                            "twfe_clustered" => function(panel::UnbalancedPanel)
                                est_for_sim(panel, 
                                    panel -> twfe_cohort_outcome_mean_ests(panel, target_cohort, target_outcome))
                                
                            end,
                            "apm_rank_1" => function(panel::UnbalancedPanel)
                                est_for_sim(panel, panel ->
                                    apm_cohort_outcome_mean_ests(panel, 1, PCCohortFactorEstimator(1), 
                                        target_cohort, target_outcome))
                            end
                        )

                        leave_out_panel_resampler = BayesianBootstrapDataGeneratingProcess(leave_out_panel)
                        boot_sim_ests = get_sim_ests(leave_out_panel_resampler, estimators, 100)
                        
                        select!(boot_sim_ests, Not(:parameter))
                        rename!(boot_sim_ests, :sim_estimate => :boot_outcome_mean)
                        boot_sim_ests[!, :cohort_str] .= target_cohort_str
                        boot_sim_ests[!, :cohort_size] .= target_cohort_size
                        boot_sim_ests[!, panel.time_col] .= target_outcome
                        boot_sim_ests[!, :true_outcome_mean] .= obs_cohort_outcome_mean_lookup[target_cohort][target_outcome]

                        append!(all_boot_sim_ests, boot_sim_ests)

                        write_parquet(cohort_outcome_file_path, boot_sim_ests)

                        outcomes_simulated_for_cohort += 1

                        GC.gc()
                    end
                end
            end

            if outcomes_simulated_for_cohort > 0
                cohort_count += 1
            end
        end
    end
end

main()
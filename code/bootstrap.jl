using Random, DataFrames, ProgressMeter, StatsBase, Distributions

include("./panel_struct.jl")

abstract type DataGeneratingProcess end

struct BayesianBootstrapDataGeneratingProcess <: DataGeneratingProcess
    panel::UnbalancedPanel
end

function get_outcome_col(boot_dgp::BayesianBootstrapDataGeneratingProcess)
    panel.time_col
end

function get_outcome_idx_map(boot_dgp::BayesianBootstrapDataGeneratingProcess)
    boot_dgp.panel.time_idx_map
end

function resample(resampler::BayesianBootstrapDataGeneratingProcess)
    UnbalancedPanel(resampler.panel, function(unit_ids)
        unnormalized_weights = rand(Exponential(), Base.length(unit_ids))
        unnormalized_weights ./ sum(unnormalized_weights)
    end)
end

function get_sim_ests(resampler::DataGeneratingProcess, estimator::Function, B = 100; multithread = true, verbose = true)
    sim_ests = nothing

    if verbose
        progress = Progress(B)
    end

    # TODO: find less verbose way of conditionally multithreading this loop
    if multithread
        data_lock = ReentrantLock()
        Threads.@threads for b in 1:B
            resampled_data = resample(resampler)
            est_results = estimator(resampled_data)
            est_results.b .= b
            rename!(est_results, :estimate => :sim_estimate)
            
            lock(data_lock) do
                if isnothing(sim_ests)
                    sim_ests = est_results
                else
                    append!(sim_ests, est_results)
                end
            end
    
            if verbose
                next!(progress)
            end
        end
    else
        for b in 1:B
            resampled_data = resample(resampler)
            est_results = estimator(resampled_data)
            est_results.b .= b
            rename!(est_results, :estimate => :sim_estimate)
        
            if isnothing(sim_ests)
                sim_ests = est_results
            else
                append!(sim_ests, est_results)
            end
    
            if verbose
                next!(progress)
            end
        end
    end

    if verbose
        finish!(progress)
    end
    
    sim_ests
end

function bayesian_bootstrap(panel::UnbalancedPanel, estimator::Function, B = 100; multithread = true, verbose = true)
    orig_ests = estimator(panel)
    rename!(orig_ests, :estimate => :orig_estimate)

    resampler = BayesianBootstrapDataGeneratingProcess(panel)
    boot_sim_ests = get_sim_ests(resampler, estimator, B, multithread = multithread, verbose = verbose)
    rename!(boot_sim_ests, :sim_estimate => :boot_estimate)
    subset!(boot_sim_ests, :boot_estimate => b -> .!isnan.(b) .&& .!ismissing.(b))
    leftjoin!(boot_sim_ests, orig_ests, on = :parameter)

    N = size(panel)[1]
    transform!(boot_sim_ests, [:boot_estimate, :orig_estimate] => 
        ((b, o) -> sqrt(N) .* (b .- o)) => :root)
    # println(combine(groupby(boot_sim_ests, :parameter), :boot_estimate => std => :boot_std_err))

    sigma_ests = combine(groupby(boot_sim_ests, :parameter), :root => 
    (r -> (quantile(r, 0.75) - quantile(r, 0.25)) 
        / (quantile(Normal(), 0.75) - quantile(Normal(), 0.25))) => :sigma)
    leftjoin!(boot_sim_ests, sigma_ests, on = :parameter)
    transform!(boot_sim_ests, [:root, :sigma] => 
        (function(r, s)
            raw_abs_t_stat = abs.(r) ./ s
            ifelse.(isnan.(raw_abs_t_stat), 0, raw_abs_t_stat)
        end) => :abs_t_stat)

    crit_vals = combine(groupby(boot_sim_ests, :parameter), :abs_t_stat => (b -> quantile(b, 0.95)) => :crit_val)
    sup_t_stats = combine(groupby(boot_sim_ests, :b), :abs_t_stat => maximum => :sup_t_stat)
    sup_crit_val = quantile(sup_t_stats.sup_t_stat, 0.95)

    final_ests = transform!(
        leftjoin(leftjoin(orig_ests, sigma_ests, on = :parameter), crit_vals, on = :parameter), 
        [:orig_estimate, :crit_val, :sigma] => 
            ((o, c, s) -> hcat((o .- c .* s ./ sqrt(N)), (o .+ c .* s ./ sqrt(N)))) => [:ci_lower, :ci_upper],
        [:orig_estimate, :sigma] => 
            ((o, s) -> hcat((o .- sup_crit_val .* s ./ sqrt(N)), (o .+ sup_crit_val .* s ./ sqrt(N)))) => 
            [:sup_cb_lower, :sup_cb_upper])
    dropmissing!(select!(final_ests, :parameter, :orig_estimate => :point_est, 
        :ci_lower, :ci_upper, :sup_cb_lower, :sup_cb_upper, :sigma, :crit_val))
end
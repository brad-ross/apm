include("./panel_struct.jl")

using LinearAlgebra, Statistics, StatsBase, FixedEffectModels

abstract type CohortFactorEstimator end

struct PCCohortFactorEstimator <: CohortFactorEstimator
    r::Int64
end

function est_cohort_factors(panel_cohort::UnbalancedPanelCohort{T}, estimator::PCCohortFactorEstimator) where T
    outcome_mat = get_cohort_outcome_mat(panel_cohort)
    unit_weights = get_cohort_unit_weight_vec(panel_cohort)
    outcome_covar_mat = outcome_mat' * (unit_weights .* outcome_mat)
    outcome_covar_mat_svd = svd(outcome_covar_mat)

    factor_ests = zeros(panel_size(panel_cohort)[2], estimator.r)
    for (cohort_idx, panel_idx) in enumerate(panel_cohort.cohort_time_idx_map)
        factor_ests[panel_idx, :] = outcome_covar_mat_svd.U[cohort_idx, 1:estimator.r]
    end

    factor_ests
end

struct DemeanedOutcomeCohortFactorEstimator <: CohortFactorEstimator
    cohort_factor_estimator::CohortFactorEstimator
end

function est_cohort_factors(panel_cohort::UnbalancedPanelCohort{T}, demaned_outcome_estimator::DemeanedOutcomeCohortFactorEstimator) where T
    est_cohort_factors(UnbalancedPanelCohort(panel_cohort, true), demaned_outcome_estimator.cohort_factor_estimator)
end

function get_cohort_time_diag_mat(panel_cohort::UnbalancedPanelCohort{T}, leave_out_times::Set{T} = Set{T}()) where T
    num_times = panel_size(panel_cohort)[2]
    output_mat_diag = zeros(num_times)
    for panel_idx in panel_cohort.cohort_time_idx_map.idx_to_id
        if id_from_idx(panel_cohort.panel_time_idx_map, panel_idx) ∉ leave_out_times
            output_mat_diag[panel_idx] = 1.
        end
    end

    Diagonal(output_mat_diag)
end

function comp_cohort_specific_statistics(panel::UnbalancedPanel, cohort_factor_estimator::CohortFactorEstimator; 
        cohorts_to_use = nothing, verbose = false)
    num_times = size(panel)[2]
    example_mat = zeros(num_times, num_times)
    cohorts = get_cohorts(panel)
    cohort_Es = create_dict_with_types_like(first(cohorts), example_mat)
    cohort_factor_ests = create_dict_with_types_like(first(cohorts), example_mat)
    cohort_outcome_mean_vecs = create_dict_with_types_like(first(cohorts), zeros(num_times))

    if verbose
        progress_bar = Progress(Base.length(cohorts))
    end
    for cohort in cohorts
        panel_cohort = UnbalancedPanelCohort(panel, cohort)
        if isnothing(cohorts_to_use) || cohort in cohorts_to_use
            cohort_factor_ests[cohort] = est_cohort_factors(panel_cohort, cohort_factor_estimator)
        end
        cohort_Es[cohort] = get_cohort_time_diag_mat(panel_cohort)

        cohort_outcome_mat = get_cohort_outcome_mat(panel_cohort)
        unit_weights = get_cohort_unit_weight_vec(panel_cohort)
        compressed_cohort_outcome_means = reshape(mean(cohort_outcome_mat, aweights(unit_weights), dims = 1), :)

        cohort_outcome_mean_vecs[cohort] = zeros(num_times)
        for cohort_outcome_idx in 1:Base.length(compressed_cohort_outcome_means)
            panel_outcome_idx = id_from_idx(panel_cohort.cohort_time_idx_map, cohort_outcome_idx)
            cohort_outcome_mean_vecs[cohort][panel_outcome_idx] = compressed_cohort_outcome_means[cohort_outcome_idx]
        end
        
        if verbose
            next!(progress_bar)
        end
    end
    if verbose
        finish!(progress_bar)
    end

    cohort_Es, cohort_factor_ests, cohort_outcome_mean_vecs
end

proj_mat(M) = M*pinv(M'*M)*M'
orthog_proj_mat(M) = I - proj_mat(M)

function comp_apm(cohort_Es::Dict{Set{S}, Matrix{T}}, cohort_factor_ests::Dict{Set{S}, Matrix{T}}) where {S, T}
    cohorts = keys(cohort_factor_ests)
    num_times = Base.size(first(values(cohort_Es)))[1]
    agg_proj_mat = zeros(num_times, num_times)

    for cohort in cohorts
        E_c = cohort_Es[cohort]
        factor_ests = cohort_factor_ests[cohort]
        agg_proj_mat += E_c * orthog_proj_mat(factor_ests) * E_c
    end

    Symmetric(agg_proj_mat)
end

function comp_apm(cohort_Es::Dict{Set{S}, Matrix{T}}, factor_ests::Matrix{T}) where {S, T}
    cohort_factor_ests = Dict{Set{S}, Matrix{T}}(
        cohort => cohort_Es[cohort] * orthog_proj_mat(cohort_Es[cohort] * factor_ests) * cohort_Es[cohort] 
        for cohort in keys(cohort_Es)
    )
    comp_apm(cohort_Es, cohort_factor_ests)
end

function comp_factor_ests_from_apm(cohort_Es::Dict{Set{S}, Matrix{T}}, cohort_factor_ests::Dict{Set{S}, Matrix{T}}, r::Int64) where {S, T}
    apm = comp_apm(cohort_Es, cohort_factor_ests)
    apm_eigen = eigen(apm)
    apm_eigen.vectors[:, 1:r]
end

function comp_cohort_mean_outcome_ests_apm(panel::UnbalancedPanel, 
                                            cohort_Es::Dict{Set{S}, Matrix{T}}, factor_ests::Matrix{T}, cohort_outcome_mean_vecs::Dict{Set{S}, Vector{T}}, 
                                            target_cohort = nothing, target_time = nothing; verbose = false) where {S, T}
    focal_cohorts = isnothing(target_cohort) ? get_cohorts(panel) : [target_cohort]
    focal_outcomes = (isnothing(target_time) ? 
        enumerate(panel.time_idx_map) : 
        [(idx_from_id(panel.time_idx_map, target_time), target_time)])

    time_id_type_arr = empty_id_arr(panel.time_idx_map)
    cohort_outcome_mean_ests = DataFrame(cohort = empty([Set(time_id_type_arr)]), outcome_mean = Float64[])
    cohort_outcome_mean_ests[!, panel.time_col] = time_id_type_arr
    select!(cohort_outcome_mean_ests, :cohort, panel.time_col, :outcome_mean)
    for cohort in focal_cohorts
            E_c = cohort_Es[cohort]
            B = (E_c * factor_ests)' \ factor_ests'
            cohort_outcome_mean_est_vec = B' * cohort_outcome_mean_vecs[cohort]
            for (time_idx, time_id) in focal_outcomes
                push!(cohort_outcome_mean_ests, (cohort, time_id, cohort_outcome_mean_est_vec[time_idx]))
            end
    end
    
    cohort_outcome_mean_ests
end

function apm_cohort_outcome_mean_ests(panel::UnbalancedPanel, r::Int64, 
        cohort_factor_estimator::CohortFactorEstimator, 
        target_cohort = nothing, target_time = nothing; 
        cohorts_to_use = nothing, verbose = false)
    if verbose
        @info "Computing cohort-specific statistics"
    end
    cohort_Es, cohort_factor_ests, cohort_outcome_mean_vecs = comp_cohort_specific_statistics(
        panel, cohort_factor_estimator, cohorts_to_use = cohorts_to_use, verbose = verbose)

    if verbose
        @info "Computing factor estimates from APM"
    end
    factor_ests = comp_factor_ests_from_apm(cohort_Es, cohort_factor_ests, r)

    if verbose
        @info "Computing cohort outcome mean estimates"
    end
    comp_cohort_mean_outcome_ests_apm(panel, cohort_Es, factor_ests, cohort_outcome_mean_vecs, 
                                    target_cohort, target_time)
end

function twfe_cohort_outcome_mean_ests(panel::UnbalancedPanel, target_cohort = nothing, target_time = nothing; 
        residual_mean_estimator::Union{Function, Nothing} = nothing)
    data_df_with_weights = innerjoin(panel.data_df, panel.unit_metadata, on = panel.unit_col)
    twfe_reg = reg(data_df_with_weights, term(panel.value_col) ~ fe(panel.unit_col) + fe(panel.time_col), 
        weights = panel.weight_col, save = true, progress_bar = false)
    all_fes = hcat(select(panel.data_df, panel.unit_col, panel.time_col), fe(twfe_reg))
    unit_fes = unique(select(all_fes, panel.unit_col, Symbol("fe_$(panel.unit_col)")))
    time_fes = unique(select(all_fes, panel.time_col, Symbol("fe_$(panel.time_col)")))
    
    if !isnothing(residual_mean_estimator)
        residuals_df = hcat(select(panel.data_df, panel.unit_col, panel.time_col), 
            DataFrame(residual = residuals(twfe_reg)))
        dropmissing!(residuals_df)
        resid_factor_means = residual_mean_estimator(residuals_df)
    end

    time_id_type_arr = empty_id_arr(panel.time_idx_map)
    cohort_outcome_mean_ests = DataFrame(cohort = Set(time_id_type_arr), outcome_mean = Float64[])
    cohort_outcome_mean_ests[!, panel.time_col] = time_id_type_arr
    select!(cohort_outcome_mean_ests, :cohort, panel.time_col, :outcome_mean)
    
    for cohort in get_cohorts(panel)
        if isnothing(target_cohort) || cohort == target_cohort
            cohort_unit_fes = leftjoin!(
                subset(panel.unit_metadata, :cohort => c -> c .== Ref(cohort)),
                unit_fes,
                on = panel.unit_col
            )
            cohort_fe_mean = mean(cohort_unit_fes[!, Symbol("fe_$(panel.unit_col)")], 
                aweights(cohort_unit_fes[!, panel.weight_col]))
            for time_fe_row in eachrow(time_fes)
                time_id, time_fe = time_fe_row[panel.time_col], time_fe_row[Symbol("fe_$(panel.time_col)")]
                if isnothing(target_time) || time_id == target_time
                    push!(cohort_outcome_mean_ests, (cohort, time_id, cohort_fe_mean + time_fe))
                end
            end
        end
    end

    if !isnothing(residual_mean_estimator)
        cohort_outcome_mean_ests = innerjoin(
            cohort_outcome_mean_ests, resid_factor_means, on = [:cohort, panel.time_col]
        )
        transform!(cohort_outcome_mean_ests, [:outcome_mean, :residual_mean] => ((o, r) -> o .+ r) => :outcome_mean)
        select!(cohort_outcome_mean_ests, Not(:residual_mean))
    end

    cohort_outcome_mean_ests
end

function twfe_cohort_outcome_mean_ests_with_factor_resids(panel::UnbalancedPanel, 
        r::Int64, cohort_factor_estimator::CohortFactorEstimator, 
        target_cohort = nothing, target_time = nothing)
    function resid_factor_means(residuals_df::DataFrame)
        residuals_df = innerjoin(residuals_df, panel.unit_metadata, on = panel.unit_col)
        resids_panel = UnbalancedPanel(residuals_df, panel.unit_col, panel.time_col, :residual, 
            panel.min_num_outcomes_per_cohort, panel.min_cohort_size,
            weight_col = panel.weight_col)
        rename!(
            apm_cohort_outcome_mean_ests(resids_panel, r, cohort_factor_estimator, target_cohort, target_time),
            :outcome_mean => :residual_mean
        )
    end

    twfe_cohort_outcome_mean_ests(panel, target_cohort, target_time, residual_mean_estimator = resid_factor_means)
end

const TEST_TOL = 1e-13

function test_apm_cohort_outcome_mean_ests_PC(include_intercepts = false)
    r = 3
    T = 2 + 2*r + 1
    cohort_size = 1000

    factors = randn(T, r)
    intercepts = include_intercepts*randn(T)
    outcome_mat = intercepts' .+ randn(3*cohort_size, r) * factors'
    full_outcome_df = DataFrame(i = Int64[], t = Int64[], y = Float64[])
    for idx in CartesianIndices(outcome_mat)
        i, t = idx[1], idx[2]
        push!(full_outcome_df, (i, t, outcome_mat[i, t]))
    end
    
    panel_df = subset(full_outcome_df, [:i, :t] => ((i, t) -> 
        (i .<= cohort_size .&& t .<= 1 + r) 
        .|| (cohort_size .< i .<= 2*cohort_size .&& 2 .<= t .<= 2 + 2*r) 
        .|| (2*cohort_size .< i .<= 3*cohort_size .&& 1 + r + 1 .<= t .<= T)))
    panel = UnbalancedPanel(panel_df, :i, :t, :y, r + 1, cohort_size)
    leftjoin!(full_outcome_df, panel.unit_metadata, on = :i)

    cohort_factor_estimator = PCCohortFactorEstimator(r)
    
    cohort_Es, cohort_factor_ests, cohort_outcome_mean_vecs = comp_cohort_specific_statistics(panel, cohort_factor_estimator)
    factor_ests = comp_factor_ests_from_apm(cohort_Es, cohort_factor_ests, r)
    @assert maximum(abs.(proj_mat(factor_ests) .- proj_mat(factors))) < TEST_TOL

    cohort_outcome_mean_ests = apm_cohort_outcome_mean_ests(panel, r, cohort_factor_estimator)
    
    time_id_type_arr = empty_id_arr(panel.time_idx_map)
    cohort_outcome_means = DataFrame(cohort = Set(time_id_type_arr), true_outcome_mean = time_id_type_arr)
    for cohort in get_cohorts(panel)
        cohort_outcome_means = combine(
            groupby(subset(full_outcome_df, :cohort => c -> c .== Ref(cohort)), :t), 
            :y => mean => :true_outcome_mean
        )
        cohort_outcome_means[!, :cohort] .= Ref(cohort)
        select!(cohort_outcome_means, :cohort, :t, :true_outcome_mean)
        append!(cohort_outcome_means, cohort_outcome_means)
    end

    errs = transform!(
        leftjoin(cohort_outcome_means, cohort_outcome_mean_ests, on = [:cohort, :t]), 
        [:outcome_mean, :true_outcome_mean] => ((e, t) -> abs.(t .- e)) => :abs_err
    )
    @assert maximum(errs.abs_err) < TEST_TOL
end

function test_twfe_cohort_outcome_mean_ests()
    time_fes = randn(8)
    cohort_size = 1000
    outcome_mat = randn(3*cohort_size) .+ time_fes'
    full_outcome_df = DataFrame(i = Int64[], t = Int64[], y = Float64[])
    for idx in CartesianIndices(outcome_mat)
        i, t = idx[1], idx[2]
        push!(full_outcome_df, (i, t, outcome_mat[i, t]))
    end
    
    panel_df = subset(full_outcome_df, [:i, :t] => ((i, t) -> 
        (i .<= cohort_size .&& t .<= 4) 
        .|| (cohort_size .< i .<= 2*cohort_size .&& 2 .<= t .<= 7) 
        .|| (2*cohort_size .< i .<= 3*cohort_size .&& 5 .<= t .<= 8)))
    panel = UnbalancedPanel(panel_df, :i, :t, :y, 4, cohort_size)
    leftjoin!(full_outcome_df, panel.unit_metadata, on = :i)
    
    cohort_outcome_mean_ests = twfe_cohort_outcome_mean_ests(panel)
    
    time_id_type_arr = empty_id_arr(panel.time_idx_map)
    cohort_outcome_means = DataFrame(cohort = Set(time_id_type_arr), true_outcome_mean = time_id_type_arr)
    for cohort in get_cohorts(panel)
        cohort_outcome_means = combine(
            groupby(subset(full_outcome_df, :cohort => c -> c .== Ref(cohort)), :t), 
            :y => mean => :true_outcome_mean
        )
        cohort_outcome_means[!, :cohort] .= Ref(cohort)
        select!(cohort_outcome_means, :cohort, :t, :true_outcome_mean)
        append!(cohort_outcome_means, cohort_outcome_means)
    end

    errs = transform!(
        leftjoin(cohort_outcome_means, cohort_outcome_mean_ests, on = [:cohort, :t]), 
        [:outcome_mean, :true_outcome_mean] => ((e, t) -> abs.(t .- e)) => :abs_err
    )
    @assert maximum(errs.abs_err) < TEST_TOL
end

function test_estimators()
    test_apm_cohort_outcome_mean_ests_PC()
    test_twfe_cohort_outcome_mean_ests()
end

test_estimators()
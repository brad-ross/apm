using DataFrames, Graphs, StatsBase, DataStructures

struct IdxIdMap{T}
    idx_to_id::Vector{T}
    id_to_idx::Dict{T, Int64}

    function IdxIdMap(ids::Vector{T}) where T
        idx_to_id = sort(unique(ids))
        id_to_idx = Dict(id => i for (i, id) in Base.enumerate(idx_to_id))
        new{T}(idx_to_id, id_to_idx)
    end
end

function idx_from_id(idx_id_map::IdxIdMap{T}, id::T) where T
    idx_id_map.id_to_idx[id]
end

function id_from_idx(idx_id_map::IdxIdMap{T}, idx::Int64) where T
    idx_id_map.idx_to_id[idx]
end

function length(idx_id_map::IdxIdMap{T}) where T
    Base.length(idx_id_map.idx_to_id)
end

function get_ids(idx_id_map::IdxIdMap{T}) where T
    idx_id_map.idx_to_id
end

function enumerate(idx_id_map::IdxIdMap{T}) where T
    Base.enumerate(idx_id_map.idx_to_id)
end

function empty_id_arr(idx_id_map::IdxIdMap{T}) where T
    similar(idx_id_map.idx_to_id, (0,))
end

function cohort_sizes_from_cohort_df(unit_metadata)
    combine(groupby(unit_metadata, :cohort), nrow => :size)
end

function get_unit_cohorts_from_data_df(data_df, unit_col, outcome_col)
    combine(groupby(data_df, unit_col), outcome_col => (t -> Set(t)) => :cohort)
end

struct UnbalancedPanel
    unit_col::Symbol
    time_col::Symbol
    value_col::Symbol
    weight_col::Symbol

    min_num_outcomes_per_cohort::Int64
    min_cohort_size::Int64

    data_df::DataFrame
    unit_metadata::DataFrame

    unit_idx_map::IdxIdMap
    time_idx_map::IdxIdMap

    function UnbalancedPanel(data_df, unit_col, time_col, value_col, 
                            min_num_outcomes_per_cohort, min_cohort_size; 
                            unit_weights = nothing, weight_col = nothing)
        cohorts_in_data_df = "cohort" ∈ names(data_df) && eltype(data_df[!, :cohort]) <: Set
        weights_in_data_df = !isnothing(weight_col) && string(weight_col) ∈ names(data_df)
        if cohorts_in_data_df || weights_in_data_df
            metadata_cols = []
            if cohorts_in_data_df
                push!(metadata_cols, :cohort)
            end
            if weights_in_data_df
                push!(metadata_cols, weight_col)
            end
            unit_metadata = unique(select(data_df, unit_col, metadata_cols))
            data_df = data_df[!, Not(metadata_cols)]
        else
            unit_metadata = get_unit_cohorts_from_data_df(data_df, unit_col, time_col)
        end
        subset!(unit_metadata, :cohort => c -> Base.length.(c) .>= min_num_outcomes_per_cohort)
        
        cohort_sizes = cohort_sizes_from_cohort_df(unit_metadata)
        subset!(cohort_sizes, :size => s -> s .>= min_cohort_size)
        select!(cohort_sizes, :cohort)
        unit_metadata = innerjoin(unit_metadata, cohort_sizes, on = :cohort)

        if isnothing(unit_weights) && isnothing(weight_col)
            weight_col = :weight
            unique_units = unique(unit_metadata[!, unit_col])
            unit_weights = DataFrame(unit_col => unique_units, weight_col => 1/Base.length(unique_units))
            unit_metadata = innerjoin(unit_metadata, unit_weights, on = unit_col)
        elseif !isnothing(unit_weights)
            if isnothing(weight_col)
                potential_weight_cols = setdiff(names(unit_weights), [string(unit_col)])
                if Base.length(potential_weight_cols) == 0
                    @error "unit_weights has no potential weight columns"
                end

                weight_col = Symbol(first(potential_weight_cols))
                if Base.length(potential_weight_cols) > 1
                    @warn "unit_weights has multiple potential weight columns; choosing the first one: $weight_col"
                end
            end

            unit_metadata = innerjoin(unit_metadata, select(unit_weights, unit_col, weight_col), on = unit_col)
        end
        
        sub_data_df = innerjoin(data_df, unique(select(unit_metadata, unit_col)), on = unit_col)
        
        unit_idx_map = IdxIdMap(sub_data_df[!, unit_col])
        time_idx_map = IdxIdMap(sub_data_df[!, time_col])
        new(unit_col, time_col, value_col, weight_col,
            min_num_outcomes_per_cohort, min_cohort_size, 
            sub_data_df, unit_metadata, unit_idx_map, time_idx_map)
    end

    function UnbalancedPanel(panel::UnbalancedPanel, new_weight_fn::Function = nothing)
        new_metadata = panel.unit_metadata
        if !isnothing(new_weight_fn)
            new_metadata = transform(panel.unit_metadata, 
                panel.unit_col => new_weight_fn => panel.weight_col)
        end

        new(panel.unit_col, panel.time_col, panel.value_col, panel.weight_col,
            panel.min_num_outcomes_per_cohort, panel.min_cohort_size, 
            panel.data_df, new_metadata, panel.unit_idx_map, panel.time_idx_map)
    end
end

function size(panel::UnbalancedPanel)
    (length(panel.unit_idx_map), length(panel.time_idx_map))
end

function get_cohorts(panel::UnbalancedPanel)
    unique(panel.unit_metadata[!, :cohort])
end

function get_unit_weights(panel::UnbalancedPanel)
    panel.unit_metadata[!, [panel.unit_col, panel.weight_col]]
end

function get_cohort_sizes(panel::UnbalancedPanel)
    sort!(cohort_sizes_from_cohort_df(panel.unit_metadata), :size, rev=true)
end

get_outcomes(panel::UnbalancedPanel) = get_ids(panel.time_idx_map)

function get_cohort_overlap_graph(cohorts::Vector{Set{T}}, r::Int) where T
    cohort_overlap_graph = SimpleGraph(Base.length(cohorts))
    for src_idx in 1:Base.length(cohorts)
        src = cohorts[src_idx]
        for dst_idx in (src_idx + 1):Base.length(cohorts)
            dst = cohorts[dst_idx]
            if Base.length(intersect(src, dst)) >= r
                add_edge!(cohort_overlap_graph, src_idx, dst_idx)
            end
        end
    end

    cohort_overlap_graph
end

function get_identified_outcomes(panel::UnbalancedPanel, r::Int)
    cohorts = unique(get_unit_cohorts_from_data_df(panel.data_df, panel.unit_col, panel.time_col)[!, :cohort])
    cohort_overlap_graph = get_cohort_overlap_graph(cohorts, r)
    cohort_overlap_graph_cc = connected_components(cohort_overlap_graph)
    sort!(cohort_overlap_graph_cc, by = Base.length, rev = true) # sort connected components in decreasing size order
    union((cohorts[cohort_idx] for cohort_idx in cohort_overlap_graph_cc[1])...)
end

create_dict_with_types_like(key_example::S, value_example::T) where {S, T} = Dict{S, T}()

function get_cohort_overlap_graph_outcome_distances(panel::UnbalancedPanel, r::Int)
    cohorts = unique(get_unit_cohorts_from_data_df(panel.data_df, panel.unit_col, panel.time_col)[!, :cohort])
    outcomes = get_outcomes(panel)
    cohort_overlap_graph = get_cohort_overlap_graph(cohorts, r)

    cohort_overlap_graph_outcome_distances = DataFrame(
        cohort = empty(cohorts), outcome = empty(outcomes), distance = Int[]
    )
    for focal_cohort_idx in vertices(cohort_overlap_graph)
        focal_cohort = cohorts[focal_cohort_idx]
        outcome_distances = create_dict_with_types_like(first(outcomes), 0)
        for outcome in outcomes
            outcome_distances[outcome] = typemax(Int)
        end

        visited = Set{Int}()
        outcomes_to_visit = Set(outcomes)
        pq = PriorityQueue{Int, Int}()
        enqueue!(pq, focal_cohort_idx, 0)
        while !isempty(pq) && !isempty(outcomes_to_visit)
            cohort_idx, dist = peek(pq)
            dequeue!(pq)

            cohort = cohorts[cohort_idx]
            setdiff!(outcomes_to_visit, cohort)
            for outcome in cohort
                outcome_distances[outcome] = min(outcome_distances[outcome], dist)
            end

            for neighbor_idx in neighbors(cohort_overlap_graph, cohort_idx)
                neighbor_idx in visited && continue
                push!(visited, neighbor_idx)
                enqueue!(pq, neighbor_idx, dist + 1)
            end
        end

        for (outcome, dist) in outcome_distances
            push!(cohort_overlap_graph_outcome_distances, (focal_cohort, outcome, dist))
        end
    end

    cohort_overlap_graph_outcome_distances
end

function constr_panel_from_identified_outcomes(panel::UnbalancedPanel, r::Int)
    identified_outcomes = get_identified_outcomes(panel, r)
    new_data_df = subset!(panel.data_df, panel.time_col => t -> t .∈ Ref(identified_outcomes))
    UnbalancedPanel(new_data_df, panel.unit_col, panel.time_col, panel.value_col, 
        panel.min_num_outcomes_per_cohort, panel.min_cohort_size, 
        unit_weights = select(panel.unit_metadata, panel.unit_col, panel.weight_col))
end

function constr_panel_with_left_out_cohort_outcome(panel::UnbalancedPanel, left_out_cohort::Set{T}, left_out_outcome::String) where T
    # by joining on the metadata before creating a new UnbalancedPanel, we maintain the same cohort structure
    data_df_with_metadata = innerjoin(panel.data_df, panel.unit_metadata, on = panel.unit_col)
    subset!(data_df_with_metadata, 
        [:cohort, panel.time_col] => ((c, t) -> c .!= Ref(left_out_cohort) .|| t .!= Ref(left_out_outcome)))
    UnbalancedPanel(data_df_with_metadata, panel.unit_col, panel.time_col, panel.value_col, 
        panel.min_num_outcomes_per_cohort, panel.min_cohort_size, 
        weight_col = panel.weight_col)
end

struct UnbalancedPanelCohort{T}
    unit_col::Symbol
    time_col::Symbol
    value_col::Symbol
    weight_col::Symbol

    cohort_times::Set{T}

    cohort_data_df::DataFrame
    cohort_unit_metadata::DataFrame

    cohort_unit_idx_map::IdxIdMap
    panel_unit_idx_map::IdxIdMap

    cohort_time_idx_map::IdxIdMap
    panel_time_idx_map::IdxIdMap

    function UnbalancedPanelCohort(panel::UnbalancedPanel, cohort_times::Set{T}) where T
        cohort_units = select!(subset(panel.unit_metadata, :cohort => c -> c .== Ref(cohort_times)), panel.unit_col)
        cohort_data_df = innerjoin(cohort_units, panel.data_df, on = panel.unit_col)
        cohort_unit_metadata = innerjoin(cohort_units, panel.unit_metadata, on = panel.unit_col)

        cohort_unit_idx_map = IdxIdMap([idx_from_id(panel.unit_idx_map, unit_id) 
            for unit_id in cohort_data_df[!, panel.unit_col]])
        cohort_time_idx_map = IdxIdMap([idx_from_id(panel.time_idx_map, time_id) 
            for time_id in cohort_data_df[!, panel.time_col]])

        new{T}(panel.unit_col, panel.time_col, panel.value_col, panel.weight_col,
            Set(unique(cohort_data_df[!, panel.time_col])), cohort_data_df, cohort_unit_metadata,
            cohort_unit_idx_map, panel.unit_idx_map, 
            cohort_time_idx_map, panel.time_idx_map)
    end

    function UnbalancedPanelCohort(panel_cohort::UnbalancedPanelCohort{T}, demean_outcomes = false) where T
        panel_cohort_data_df = panel_cohort.cohort_data_df
        if demean_outcomes
            panel_cohort_data_df_with_metadata = innerjoin(panel_cohort_data_df, panel_cohort.cohort_unit_metadata, 
                on = panel_cohort.unit_col)
            cohort_outcome_means = combine(
                groupby(panel_cohort_data_df_with_metadata, panel_cohort.time_col), 
                [panel_cohort.value_col, panel_cohort.weight_col] => ((v, w) -> mean(v, aweights(w))) => :outcome_mean)
            panel_cohort_data_df = select!(
                innerjoin(panel_cohort_data_df, cohort_outcome_means, on = panel_cohort.time_col),
                panel_cohort.unit_col, panel_cohort.time_col,
                [panel_cohort.value_col, :outcome_mean] => ((v, m) -> v .- m) => panel_cohort.value_col
            )
        end

        new{T}(panel_cohort.unit_col, panel_cohort.time_col, panel_cohort.value_col, panel_cohort.weight_col,
            panel_cohort.cohort_times, panel_cohort_data_df, panel_cohort.cohort_unit_metadata,
            panel_cohort.cohort_unit_idx_map, panel_cohort.panel_unit_idx_map, 
            panel_cohort.cohort_time_idx_map, panel_cohort.panel_time_idx_map)
    end
end

function size(panel_cohort::UnbalancedPanelCohort)
    (length(panel_cohort.cohort_unit_idx_map), length(panel_cohort.cohort_time_idx_map))
end

function panel_size(panel_cohort::UnbalancedPanelCohort)
    (length(panel_cohort.panel_unit_idx_map), length(panel_cohort.panel_time_idx_map))
end

function get_cohort_outcome_mat(panel_cohort::UnbalancedPanelCohort)
    array = NaN*zeros(size(panel_cohort))
    for row in eachrow(panel_cohort.cohort_data_df)
        i = idx_from_id(panel_cohort.cohort_unit_idx_map, 
            idx_from_id(panel_cohort.panel_unit_idx_map, row[panel_cohort.unit_col]))
        t = idx_from_id(panel_cohort.cohort_time_idx_map, 
            idx_from_id(panel_cohort.panel_time_idx_map, row[panel_cohort.time_col]))
        array[i, t] = row[panel_cohort.value_col]
    end

    array
end

function get_cohort_unit_weight_vec(panel_cohort::UnbalancedPanelCohort)
    vec = NaN*zeros(size(panel_cohort)[1])
    for row in eachrow(panel_cohort.cohort_unit_metadata)
        i = idx_from_id(panel_cohort.cohort_unit_idx_map, 
            idx_from_id(panel_cohort.panel_unit_idx_map, row[panel_cohort.unit_col]))
        vec[i] = row[panel_cohort.weight_col]
    end

    vec
end
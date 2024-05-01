using Chain, DataFrames, Parquet, Statistics, Clustering, Random, ProgressMeter

include("code/vwh_data_helpers.jl")
include("code/panel_struct.jl")

const RAW_DATA_PATH = joinpath(ENV["DATA_PATH"], "raw_data")
const CLEAN_DATA_PATH = joinpath(ENV["DATA_PATH"], "clean_data")

const OUTPUT_PATH = ENV["OUTPUT_PATH"]

Random.seed!(1337)

function get_group_wage_dists(avg_weekly_earn_by_group, group_vars, earnings_quantile_grid_size = 100)
    earnings_quantile_grid = range(0.01, 0.99, length=earnings_quantile_grid_size)
    earnings_quantiles = quantile(avg_weekly_earn_by_group.avg_weekly_earnings, earnings_quantile_grid)
    @chain avg_weekly_earn_by_group begin
        groupby(group_vars)
        combine(nrow => :num_workers, :avg_weekly_earnings => 
            (e -> [earnings_quantiles mean(e' .<= earnings_quantiles, dims = 2)]) => [:wage_quantile, :wage_cdf_val])
        dropmissing!
    end
end

function cluster_groups(group_wage_dists, group_vars, wage_dist_name_var, wage_dist_val_var, num_clusters; cluster_within = nothing, maxiter = 200, num_reps = 10)
    horiz_group_wage_dists = @chain group_wage_dists begin
        select(group_vars, wage_dist_name_var, wage_dist_val_var)
        unstack(group_vars, wage_dist_name_var, wage_dist_val_var, fill = 0.0)
        innerjoin((@chain group_wage_dists begin
            select(group_vars, :num_workers)
            unique!
            groupby(group_vars)
            combine(:num_workers => sum => :num_workers)
        end), on = group_vars)
        dropmissing!
    end

    if isnothing(cluster_within)
        horiz_group_wage_dists.dummy_col .= true
        cluster_within = [:DUMMY_COL]
    end

    horiz_group_wage_dists_by_outer_cluster = groupby(horiz_group_wage_dists, cluster_within)

    group_cluster_assignments = empty(select(horiz_group_wage_dists, group_vars, cluster_within, :num_workers, 
        :num_workers => (n -> 1) => :firm_type))
    group_cluster_assignments_lock = ReentrantLock()
    num_outer_clusters = Base.length(collect(keys(horiz_group_wage_dists_by_outer_cluster)))
    progress = Progress(num_reps*num_outer_clusters)
    for horiz_group_wage_dists_within_outer_cluster in horiz_group_wage_dists_by_outer_cluster
        horiz_group_wage_dist_mat = Matrix(select(horiz_group_wage_dists_within_outer_cluster, 
            Not([group_vars; cluster_within; :num_workers])))
        group_weights = collect(horiz_group_wage_dists_within_outer_cluster.num_workers)
        best_clustering = nothing
        best_clustering_lock = ReentrantLock()

        Threads.@threads for i in 1:num_reps
            new_clustering = kmeans(horiz_group_wage_dist_mat', num_clusters, weights = group_weights, maxiter = maxiter)

            if new_clustering.iterations == maxiter
                @warn "Iteration $i: $(new_clustering.totalcost); did not converge in $(new_clustering.iterations) steps"
            end

            lock(best_clustering_lock) do
                if isnothing(best_clustering) || best_clustering.totalcost > new_clustering.totalcost
                    best_clustering = new_clustering
                end
            end

            next!(progress)
        end

        group_cluster_assignments_within_outer_cluster = select(horiz_group_wage_dists_within_outer_cluster, 
            group_vars, cluster_within, :num_workers)
        group_cluster_assignments_within_outer_cluster.firm_type = best_clustering.assignments

        lock(group_cluster_assignments_lock) do
            append!(group_cluster_assignments, group_cluster_assignments_within_outer_cluster)
        end
    end
    finish!(progress)

    group_cluster_assignments
end

function main()
    earnings = read_raw_earnings_data()
    firms = read_raw_firm_data()
    
    clustering_output_path = joinpath(CLEAN_DATA_PATH, "clustering_connectivity_exploration_provinces")

    start_years = [1992, 1996, 1998]
    year_cluster_sizes = [1, 2, 3, 5, 10]
    num_clusters_options = [1, 2, 3, 4, 5]
    
    for start_year in start_years
        println("====== Start Year: $(start_year) ======")
        raw_joined_data = get_joined_and_filtered_match_data(earnings, firms, start_year, 2001, VENETO_PROVINCES)
        start_year_output_path = joinpath(clustering_output_path, "start_year=$(start_year)")
        
        for year_cluster_size in filter(n -> n <= (2001 - start_year + 1), year_cluster_sizes)
            println("===== Year Cluster Size: $(year_cluster_size) =====")
            year_cluster_size_output_path = joinpath(start_year_output_path, "year_cluster_size=$(year_cluster_size)")
            mkpath(year_cluster_size_output_path)

            joined_data = copy(raw_joined_data)
            joined_data.year_cluster = get_year_clusters(joined_data.year, year_cluster_size)
            filter_joined_data_to_always_present_workers_and_firms!(joined_data)
        
            avg_weekly_earn_by_group = get_avg_weekly_earn_by_group(joined_data, [:year_cluster, :firm, :province])
            group_wage_dists = @chain avg_weekly_earn_by_group begin
                get_group_wage_dists([:year_cluster, :firm, :province])
                # to cluster only on first year cluster's wage distribution:
                subset!(:year_cluster => (y -> y .== start_year))
                dropmissing!
                transform!([:year_cluster, :wage_quantile] => 
                    ((y, q) -> string.(y) .* "_" .* string.(q)) => :year_cluster_wage_quantile)
            end

            for num_clusters in num_clusters_options
                println("==== # Clusters: $(num_clusters) ====")
                num_clusters_output_path = joinpath(year_cluster_size_output_path, "num_clusters=$(num_clusters)")
                mkpath(num_clusters_output_path)

                clustered_firms = cluster_groups(group_wage_dists, [:firm, :province], :year_cluster_wage_quantile, :wage_cdf_val, 
                    num_clusters, cluster_within = [:province])

                write_parquet(joinpath(num_clusters_output_path, "part-0.parquet"), clustered_firms)
            end
        end
    end

    panel_cluster_sizes = nothing
    for start_year in start_years
        println("====== Start Year: $(start_year) ======")
        raw_joined_data = get_joined_and_filtered_match_data(earnings, firms, start_year, 2001, VENETO_PROVINCES)
        start_year_output_path = joinpath(clustering_output_path, "start_year=$(start_year)")
        
        for year_cluster_size in filter(n -> n <= (2001 - start_year + 1), year_cluster_sizes)
            println("===== Year Cluster Size: $(year_cluster_size) =====")
            year_cluster_size_output_path = joinpath(start_year_output_path, "year_cluster_size=$(year_cluster_size)")

            joined_data = copy(raw_joined_data)
            joined_data.year_cluster = get_year_clusters(joined_data.year, year_cluster_size)
            filter_joined_data_to_always_present_workers_and_firms!(joined_data)

            for num_clusters in num_clusters_options
                println("==== # Clusters: $(num_clusters) ====")
                num_clusters_output_path = joinpath(year_cluster_size_output_path, "num_clusters=$(num_clusters)")
                clustered_firms = DataFrame(read_parquet(num_clusters_output_path))

                joined_data_with_clusters = @chain joined_data begin
                    leftjoin(select(clustered_firms, Not(:num_workers)), on = [:firm, :province])
                end

                avg_weekly_earn_by_type = get_avg_weekly_earn_by_group(joined_data_with_clusters, [:year_cluster, :province, :firm_type])
                transform!(avg_weekly_earn_by_type, [:year_cluster, :province, :firm_type] => 
                    ((y, p, t) -> string.(t) .* "_" .* p .* "_" .* string.(y)) => :outcome)
                num_groups_observed_dist = @chain avg_weekly_earn_by_type begin
                    groupby(:worker)
                    combine(nrow => :num_groups_observed)
                    groupby(:num_groups_observed)
                    combine(nrow => :num_workers)
                end

                for min_num_outcomes_per_cohort in [2]
                    println("=== Minimum # Outcomes/Cohort: $(min_num_outcomes_per_cohort) ===")
                    for r in filter(r -> r <= min_num_outcomes_per_cohort, [1, 2])
                        println("== Factor Model Rank: $(r) ==")
                        for min_cohort_size in [150, 500]
                            println("= Minimum Cohort Size: $(min_cohort_size) =")
                            orig_panel = UnbalancedPanel(avg_weekly_earn_by_type, :worker, :outcome, :avg_weekly_earnings, min_num_outcomes_per_cohort, min_cohort_size)
                            raw_identified_outcomes = get_identified_outcomes(orig_panel, r)
                            identified_outcomes = DataFrame(
                                (firm_type = parse(Int64, firm_type), province = province, year_cluster = parse(Int64, year_cluster))
                                for (firm_type, province, year_cluster) in
                                split.(raw_identified_outcomes, "_")
                            )

                            curr_panel_cluster_sizes = @chain avg_weekly_earn_by_type begin
                                leftjoin(select(orig_panel.data_df, :worker, :outcome, :worker => (w -> true) => :in_panel), on = [:worker, :outcome])
                                transform!(:in_panel => (i -> coalesce.(i, false)) => :in_panel)
                                groupby([:year_cluster, :province, :firm_type])
                                combine(nrow => :num_workers, :in_panel => sum => :num_workers_panel)
                                leftjoin(
                                    transform(identified_outcomes, :year_cluster => (y -> true) => :identified), 
                                    on = [:year_cluster, :province, :firm_type]
                                )
                                transform!(:identified => (i -> coalesce.(i, false)) => :identified)
                                groupby(:year_cluster)
                                transform!(
                                    [:num_workers, :num_workers_panel] => ((n, p) -> p ./ sum(n)) => :share_workers,
                                    [:num_workers, :num_workers_panel] => ((n, p) -> p ./ sum(p)) => :share_workers_panel
                                )
                                sort([:firm_type, :province, :year_cluster], rev = true)
                            end

                            curr_panel_cluster_sizes.start_year .= start_year
                            curr_panel_cluster_sizes.year_cluster_size .= year_cluster_size
                            curr_panel_cluster_sizes.num_clusters .= num_clusters
                            curr_panel_cluster_sizes.min_num_outcomes_per_cohort .= min_num_outcomes_per_cohort
                            curr_panel_cluster_sizes.r .= r
                            curr_panel_cluster_sizes.min_cohort_size .= min_cohort_size
                            select!(curr_panel_cluster_sizes, 
                                :start_year, :year_cluster_size, :num_clusters, :min_num_outcomes_per_cohort, :r, :min_cohort_size, All())

                            if isnothing(panel_cluster_sizes)
                                panel_cluster_sizes = curr_panel_cluster_sizes
                            else
                                append!(panel_cluster_sizes, curr_panel_cluster_sizes)
                            end
                        end
                    end
                end
            end
        end
    end

    clustering_exploration_output_path = joinpath(OUTPUT_PATH, "clustering_exploration_provinces")
    mkpath(clustering_exploration_output_path)
    write_parquet(joinpath(clustering_exploration_output_path, "panel_cluster_sizes_veneto.parquet"), panel_cluster_sizes)

    panel_cluster_sizes = DataFrame(read_parquet(joinpath(clustering_exploration_output_path, "panel_cluster_sizes_veneto.parquet")))
    panel_coverage_rates = @chain panel_cluster_sizes begin
        subset(:identified => identity)
        groupby([:start_year, :year_cluster_size, :num_clusters, :min_num_outcomes_per_cohort, :min_cohort_size, :r, :year_cluster])
        combine([:share_workers, :identified] => ((s, i) -> sum(s .* i)) => :coverage_rate,
            [:share_workers_panel, :identified] => ((s, i) -> sum(s .* i)) => :coverage_rate_panel,
            :num_workers => sum => :num_workers,
            :num_workers_panel => sum => :num_workers_panel,
            [:num_workers, :identified] => ((n, i) -> sum(n .* i)) => :num_identified_workers,
            [:num_workers_panel, :identified] => ((n, i) -> sum(n .* i)) => :num_identified_workers_panel,)
        groupby([:start_year, :year_cluster_size, :num_clusters, :min_num_outcomes_per_cohort, :min_cohort_size, :r])
        combine(:coverage_rate => maximum => :max_cov_rate,
            :coverage_rate => minimum => :min_cov_rate,
            :coverage_rate_panel => maximum => :max_cov_rate_panel,
            :coverage_rate_panel => minimum => :min_cov_rate_panel,
            :num_workers => maximum => :max_num_workers,
            :num_workers => minimum => :min_num_workers,
            :num_workers_panel => maximum => :max_num_workers_panel,
            :num_workers_panel => minimum => :min_num_workers_panel,
            :num_identified_workers => maximum => :max_num_id_workers,
            :num_identified_workers => minimum => :min_num_id_workers,
            :num_identified_workers_panel => maximum => :max_num_id_workers_panel,
            :num_identified_workers_panel => minimum => :min_num_id_workers_panel)
        rename!(:min_num_outcomes_per_cohort => :min_T_c, :year_cluster_size => :num_yrs)
        sort!([:start_year, :num_yrs, :num_clusters, :min_T_c, :min_cohort_size, :r])
    end
end

main()
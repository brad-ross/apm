using Chain, DataFrames, Parquet

const RAW_DATA_PATH = joinpath(ENV["DATA_PATH"], "raw_data")
const CLEAN_DATA_PATH = joinpath(ENV["DATA_PATH"], "clean_data")

const VENETO_PROVINCES = Set(["BL", "PD", "RO", "TV", "VE", "VR", "VI"])

function read_raw_earnings_data()
    raw_earnings_data = DataFrame(read_parquet(joinpath(RAW_DATA_PATH, "earnings.parquet")))
    select!(raw_earnings_data,
        :cod_pgr => (w -> convert.(Int64, w)) => :worker, :matr_az => (f -> convert.(Int64, f)) => :firm, 
        :anno => (y -> convert.(Int64, y)) => :year, 
        :sett_r => (w -> convert.(Int64, w)) => :paid_weeks, :gior_r => (d -> convert.(Int64, d)) => :paid_days,
        :retrib03 => :earnings, :prov_l => (p -> ifelse.(p .== "", missing, p)) => :province)
end

function read_raw_firm_data()
    raw_firm_data = DataFrame(read_parquet(joinpath(RAW_DATA_PATH, "firms.parquet")))
    select!(raw_firm_data,
    :matr_az => (f -> convert.(Int64, f)) => :firm, :cap => (p -> convert.(Union{Int64, Missing}, p)) => :postal_code,
    :comune => :city, :prov => :province, :csc => (c -> convert.(Union{Int64, Missing}, c)) => :taxation_code)
end

function get_joined_and_filtered_match_data(raw_earnings, raw_firms, min_year, max_year, provinces = nothing)
    @chain raw_earnings begin
        select(Not(:province))
        leftjoin(raw_firms, on = :firm, makeunique=true)
        subset!([:year, :province, :paid_weeks] => 
            (y, p, w) -> (min_year .<= y .<= max_year) .&& (isnothing(provinces) .|| (.!ismissing.(p) .&& p .∈ Ref(provinces))) .&& (w .> 0))
    end
end

function get_year_clusters(years, bin_width)
    min_year = minimum(years)
    min_year .+ convert.(Int64, floor.((years .- min_year)./bin_width) .* bin_width)
end

function filter_joined_data_to_always_present_workers_and_firms!(joined_data)
    year_clusters_per_firm = @chain joined_data begin
        select(:year_cluster, :firm)
        unique!()
        groupby(:firm)
        combine(nrow => :firm_num_year_clusters)
    end
    
    year_clusters_per_worker = @chain joined_data begin
        select(:worker, :year_cluster)
        unique!()
        groupby(:worker)
        combine(nrow => :worker_num_year_clusters)
    end

    total_year_clusters = Base.length(unique(joined_data.year_cluster))

    @chain joined_data begin
        leftjoin!(year_clusters_per_firm, on = :firm)
        leftjoin!(year_clusters_per_worker, on = :worker)
        subset!(
            :firm_num_year_clusters => (n -> n .== total_year_clusters),
            :worker_num_year_clusters => (n -> n .== total_year_clusters)
        )
    end
end

function get_avg_weekly_earn_by_group(joined_data, group_vars)
    @chain joined_data begin
        groupby([:worker; group_vars])
        combine([:paid_weeks, :earnings] => ((w, e) -> sum(e) / sum(w)) => :avg_weekly_earnings)
        dropmissing!
    end
end
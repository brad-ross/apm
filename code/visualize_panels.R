library(tidyverse)
library(arrow)
library(data.table)
library(igraph)
library(ggraph)

DATA_PATH = Sys.getenv("DATA_PATH")
RAW_DATA_PATH = file.path(DATA_PATH, "raw_data")
CLEAN_DATA_PATH = file.path(DATA_PATH, "clean_data")
OUTPUT_PATH = Sys.getenv("OUTPUT_PATH")

# worker firm data

panel_data = read_parquet(file.path(CLEAN_DATA_PATH, "final_panel.parquet")) |>
    select(-c(avg_weekly_earnings, log_avg_weekly_earnings)) |>
    mutate(raw_outcome = outcome) |>
    unite("outcome", year_cluster, province, firm_type, remove = FALSE) |>
    mutate(
        worker = as.factor(worker),
        outcome = as.factor(outcome),
        year_cluster = as.factor(year_cluster),
        province = as.factor(province),
        firm_type = as.factor(firm_type)
    ) |>
    arrange(worker, outcome) |>
    group_by(worker) |>
    mutate(
        cohort = str_c(outcome, collapse = "; ")
    ) |>
    ungroup()

unit_cohorts = panel_data |>
    select(cohort, worker) |>
    distinct()
cohort_sizes = unit_cohorts |>
    group_by(cohort) |>
    summarize(cohort_size = n()) |>
    arrange(desc(cohort_size)) |>
    mutate(cohort_id = row_number())
unit_ids = unit_cohorts |>
    inner_join(cohort_sizes, by = "cohort") |>
    arrange(cohort_size) |>
    mutate(unit_id = row_number())
cohort_rectangles = panel_data |>
    select(cohort, outcome) |>
    unique() |>
    left_join(
    unit_ids |>
        group_by(cohort, cohort_id) |>
        summarize(
            unit_id_min = min(unit_id),
            unit_id_max = max(unit_id)
        ),
        by = "cohort"
    )
panel_data_to_plot = panel_data |>
    inner_join(unit_ids, by = "worker")

panel_plot_rectanges = (
    ggplot(cohort_rectangles)
    + theme_test()
    + theme(panel.background=element_rect(fill="#f5e7e7"),
            panel.border = element_blank(),
            axis.text.x = element_text(size = 6))
    + geom_tile(aes(y = (unit_id_min + unit_id_max)/2, height = unit_id_max - unit_id_min, x = outcome), fill = "#7a0a0a")
    + scale_x_discrete(expand = c(0, 0), labels = function(x) {
        split_x = str_split_fixed(x, "_", 3)
        year_cluster = split_x[,1]
        year_cluster_label = str_glue("{year_cluster} - {as.numeric(year_cluster) + 1}")
        province = split_x[,2]
        firm_type = split_x[,3]        
        if_else(firm_type == "2", 
            if_else(province == "TV",
                str_glue("{firm_type}\n{province}\n ----- {year_cluster_label} -----"),
                str_glue("{firm_type}\n{province}")), 
            str_glue("{firm_type}"))
    })
    + scale_y_continuous(expand = c(0, 0), labels = scales::scientific)
    + labs(x = "(Firm Type, Province, Year Range)", y = "Worker")
)
ggsave(file.path(OUTPUT_PATH, "panel_plot_worker_locations.pdf"), panel_plot_rectanges, width = 3.0625, height = 3.25)

# plot cohort spectra

cohort_spectra = read_csv(file.path(OUTPUT_PATH, "cohort_sizes_and_spectra.csv"))

cohort_spectra_pairs = cohort_spectra |>
    inner_join(cohort_spectra |>
        filter(spectrum_idx > 1) |>
        mutate(prev_spectrum_idx = spectrum_idx - 1) |>
        select(cohort_str, prev_spectrum_idx, next_eigenvalue = eigenvalue),
        by = c("cohort_str" = "cohort_str", "spectrum_idx" = "prev_spectrum_idx")
    ) |>
    mutate(eigenvalue_ratio = eigenvalue / next_eigenvalue)

cohort_eigengap_ranks = cohort_spectra_pairs |>
    group_by(spectrum_idx) |>
    arrange(eigenvalue_ratio) |>
    mutate(rank = row_number(), cumu_cohort_size = cumsum(cohort_size)) |>
    ungroup() |>
    arrange(spectrum_idx, cohort_size)
cohort_eigengap_summaries = cohort_eigengap_ranks |>
    group_by(spectrum_idx) |>
    summarize(n = n(), 
        num_units = sum(cohort_size))
cohort_eigengap_dists = cohort_eigengap_ranks |>
    inner_join(cohort_eigengap_summaries, by = "spectrum_idx") |>
    mutate(eigenvalue_ratio_pctile = rank / n, eigenvalue_ratio_units_pctile = cumu_cohort_size / num_units) |>
    bind_rows(tibble(spectrum_idx = unique(cohort_eigengap_ranks$spectrum_idx), 
        eigenvalue_ratio_pctile = 0, eigenvalue_ratio_units_pctile = 0,
        eigenvalue_ratio = min(cohort_eigengap_ranks$eigenvalue_ratio) - 1e-16))

# taken from https://stackoverflow.com/questions/10762287/how-can-i-format-axis-labels-with-exponents-with-ggplot2-and-scales
scientific_10 = function(x) {
  parse(text=gsub("e\\+*", " %*% 10^", scales::scientific_format()(x)))
}

cohort_eigengap_dists_plot = (
    ggplot(cohort_eigengap_dists |> mutate(spectrum_idx = as.factor(spectrum_idx)))
    + theme_bw()
    + geom_step(aes(y = eigenvalue_ratio_units_pctile, x = eigenvalue_ratio, group = spectrum_idx, color = spectrum_idx), linewidth = 1.25)
    + scale_x_log10(
        breaks = scales::trans_breaks("log10", function(x) 10^x),
        labels = scales::trans_format("log10", scales::math_format(10^.x))
    )
    + scale_y_continuous(limits = c(0, 1))
    + scale_color_viridis_d(limits = as.factor(1:4), end = 0.9)
    + labs(y = "Share Units w/ Lower Eigval. Ratio", x = "Larger / Smaller Eigenvalue", color = "Rank of Larger Eigenvalue:")
    + theme(
        legend.position = "bottom", 
        legend.title.position = "top", 
        legend.margin = margin(-7, 0, -5, 0),
    )
)

ggsave(file.path(OUTPUT_PATH, "cohort_eigengap_dists_plot.pdf"), 
        cohort_eigengap_dists_plot, width = 8, height = 3.25)

# plot cohort connectivity per outcome

cohort_outcome_dists = read_csv(file.path(OUTPUT_PATH, "cohort_outcome_dists.csv"))

cohort_outcome_hops = cohort_outcome_dists |>
    select(cohort_str, hops = distance_rank_1) |>
    mutate(hops = as.integer(hops)) |>
    group_by(cohort_str, hops) |>
    summarize(num_outcomes_reached = n()) |>
    arrange(cohort_str, hops) |>
    mutate(cumu_outcomes_reached = cumsum(num_outcomes_reached)) |>
    ungroup()
cohort_outcome_hops = bind_rows(cohort_outcome_hops,
    cohort_outcome_hops |> 
        select(cohort_str) |> 
        distinct() |> 
        mutate(hops = max(cohort_outcome_hops$hops)) |> 
        mutate(num_outcomes_reached = 0, 
            cumu_outcomes_reached = max(cohort_outcome_hops$cumu_outcomes_reached))) |>
    arrange(cohort_str, hops)

cohort_outcome_hop_ranks = cohort_outcome_hops |>
    inner_join(cohort_spectra |> select(cohort_str, cohort_size) |> distinct(), by = "cohort_str") |>
    group_by(hops, cumu_outcomes_reached) |>
    summarize(n = n(), cohort_size = sum(cohort_size)) |>
    group_by(hops) |>
    arrange(cumu_outcomes_reached) |>
    mutate(rank = row_number(), cumu_cohort_size = cumsum(cohort_size)) |>
    ungroup() |>
    arrange(hops, cumu_outcomes_reached)
cohort_outcome_hop_summaries = cohort_outcome_hop_ranks |>
    group_by(hops) |>
    summarize(num_cohorts = sum(n), 
        num_units = sum(cohort_size))
cohort_outcome_hop_dists = cohort_outcome_hop_ranks |>
    inner_join(cohort_outcome_hop_summaries, by = "hops") |>
    mutate(cumu_outcomes_reached_pctile = rank / num_cohorts, 
        cumu_outcomes_reached_units_pctile = cumu_cohort_size / num_units) |>
    bind_rows(tibble(hops = unique(cohort_outcome_hop_ranks$hops), 
        cumu_outcomes_reached_pctile = 0, cumu_outcomes_reached_units_pctile = 0,
        cumu_outcomes_reached = min(cohort_outcome_hop_ranks$cumu_outcomes_reached) - 1e-10)) |>
    arrange(hops, cumu_outcomes_reached)

cohort_outcome_hop_dists_plot = (
    ggplot(cohort_outcome_hop_dists |> mutate(hops = as.factor(hops)))
    + theme_bw()
    + geom_step(aes(y = cumu_outcomes_reached_units_pctile, x = cumu_outcomes_reached, group = hops, color = hops), linewidth = 1.25)
    + scale_y_continuous(limits = c(0, 1))
    + scale_color_viridis_d(limits = as.factor(
        min(unique(cohort_outcome_hop_ranks$hops)):(max(unique(cohort_outcome_hop_ranks$hops)) - 1)), 
        end = 0.9)
    + labs(y = "Share w/ < # Reached Outcomes", x = "# Outcomes Reached at BFS Edge Distance", color = "BFS Edge Distance:")
    + theme(
        legend.position = "bottom", 
        legend.title.position = "top", 
        legend.margin = margin(-7, 0, -5, 0),
    )
)

ggsave(file.path(OUTPUT_PATH, "cohort_outcome_hop_dists_plot.pdf"), 
        cohort_outcome_hop_dists_plot, width = 8, height = 3.25)

# plot overlap graph

cohorts_expanded = cohort_sizes |>
    separate_wider_delim(cohort, delim = "; ", names = paste0("outcome_", c("1", "2", "3", "4", "5")), too_few = "align_start") |>
    pivot_longer(cols = starts_with("outcome"), names_to = "outcome_num", values_to = "outcome") |>
    drop_na() |>
    select(cohort_id, outcome, cohort_size)

overlap_graph_nodes = cohorts_expanded |>
    select(cohort_id, cohort_size) |>
    distinct()

overlap_graph_edges = cohorts_expanded |>
    select(-c(cohort_size)) |>
    left_join(cohorts_expanded |> select(-c(cohort_size)), 
        by = "outcome", suffix = c("_l", "_r"), relationship = "many-to-many") |>
    filter(cohort_id_l != cohort_id_r) |>
    select(cohort_id_l, cohort_id_r, outcome)

overlap_graph = graph_from_data_frame(d=overlap_graph_edges, vertices=overlap_graph_nodes, directed=FALSE)

overlap_graph_plot = (
    ggraph(overlap_graph)
    + theme_void()
    + geom_edge_link(width = 0.25, alpha = 0.15, color = "grey")
    + geom_node_point(aes(size = cohort_size), stroke = 0.25, shape = 21, fill = "#7a0a0a")
    + guides(size = "none")
)
ggsave(file.path(OUTPUT_PATH, "overlap_graph_1_plot_worker_locations.pdf"), overlap_graph_plot, width = 3.0625, height = 3.25)

# plot non-block-missing cohort outcomes

cohorts_expanded = cohort_sizes |>
    separate_wider_delim(cohort, delim = "; ", names = c("prov_1", "prov_2", "prov_3")) |>
    pivot_longer(cols = starts_with("prov"), names_to = "prov_num", values_to = "province") |>
    select(cohort_id, province)
cohorts_missing_provs = anti_join( 
    cross_join(
        cohorts_expanded |> select(cohort_id) |> distinct(),
        cohorts_expanded |> select(province) |> distinct()
    ), cohorts_expanded)
single_prov_overlaps = cohorts_expanded |>
    inner_join(cohorts_expanded, by = "province", suffix = c("_1", "_2")) |>
    filter(cohort_id_1 != cohort_id_2)
double_prov_overlaps = single_prov_overlaps |>
    inner_join(single_prov_overlaps, by = c("cohort_id_1", "cohort_id_2"), suffix = c("_1", "_2")) |>
    filter(province_1 != province_2)
block_missing_id_provs = double_prov_overlaps |>
    inner_join(cohorts_expanded, by = c("cohort_id_2" = "cohort_id")) |>
    filter(province_1 != province, province_2 != province) |>
    arrange(cohort_id_1)
block_missing_nonid_outcomes = cohorts_missing_provs |>
    left_join(block_missing_id_provs, by = c("cohort_id" = "cohort_id_1", "province")) |>
    filter(is.na(cohort_id_2)) |>
    select(cohort_id, province)

panel_cohort_outcome_to_highlight = unit_ids |>
    inner_join(block_missing_nonid_outcomes, by = "cohort_id") |>
    filter(cohort_id == 5)
panel_data_to_plot_w_missing_outcome = bind_rows(
    panel_data_to_plot |> mutate(missing_outcome = FALSE),
    panel_cohort_outcome_to_highlight |> mutate(missing_outcome = TRUE)
)

missing_cohort_outcome_panel_plot = (
    ggplot(panel_data_to_plot_w_missing_outcome)
    + theme_test()
    + geom_tile(aes(x = province, y = unit_id, fill = as.character(missing_outcome)), show.legend = FALSE)
    + theme(panel.background=element_rect(fill="#f5e7e7"),
            panel.border = element_blank())
    + scale_x_discrete(expand = c(0, 0))
    + scale_y_continuous(expand = c(0, 0))
    + scale_fill_manual(breaks = c("FALSE", "TRUE"), values = c("#7a0a0a", "#340505"))
    + labs(x = "Province", y = "Worker")
)
ggsave(file.path(OUTPUT_PATH, "panel_plot_worker_locations_missing_cohort_outcomes.pdf"), 
    missing_cohort_outcome_panel_plot, width = 3.0625, height = 3.25)

# outcomes over time data

driver_data = read_csv(file.path(RAW_DATA_PATH, "riders.csv")) |>
    select(rider_id, first_week_id = first_week_in_panel_dateweek_id, treatment_week_id = began_active_week_id) |>
    drop_na() |>
    as.data.table()
panel_data = driver_data[, 
    list(week_id = seq(first_week_id, treatment_week_id)), 
    by = c("rider_id", "first_week_id", "treatment_week_id")
] |>
    mutate(first_week_padded = str_pad(as.character(first_week_id), 2, pad = "0"),
        treatment_week_padded = str_pad(as.character(treatment_week_id), 2, pad = "0")) |>
    unite(cohort, first_week_padded, treatment_week_padded, sep = " - ") |>
    select(cohort, rider_id, first_week_id, treatment_week_id)

unit_cohorts = panel_data |>
    select(cohort, rider_id) |>
    distinct()
cohort_sizes = unit_cohorts |>
    group_by(cohort) |>
    summarize(cohort_size = n()) |>
    filter(cohort_size >= 100) |>
    arrange(desc(cohort_size)) |>
    mutate(cohort_id = row_number())
unit_ids = unit_cohorts |>
    inner_join(cohort_sizes, by = "cohort") |>
    arrange(cohort) |>
    mutate(unit_id = row_number())
panel_data_to_plot = panel_data |>
    inner_join(unit_ids, by = "rider_id")

panel_plot = (
    ggplot(panel_data_to_plot)
    + theme_test()
    + geom_tile(aes(x = week_id, y = unit_id), fill = "#7a0a0a")
    + theme(panel.background=element_rect(fill="#f5e7e7"),
            panel.border = element_blank())
    + scale_x_continuous(expand = c(0, 0))
    + scale_y_continuous(expand = c(0, 0))
    + labs(x = "Week", y = "Driver")
)
ggsave(file.path(OUTPUT_PATH, "panel_plot_drivers_times.pdf"), panel_plot, width = 3.0625, height = 3.25)

panel_data_to_plot_w_missing_outcome = bind_rows(
    panel_data_to_plot |> mutate(missing_outcome = FALSE),
    unit_ids |> filter(cohort_id == 1) |> mutate(week_id = 45) |> mutate(missing_outcome = TRUE)
)

missing_cohort_outcome_panel_plot = (
    ggplot(panel_data_to_plot_w_missing_outcome)
    + theme_test()
    + geom_tile(aes(x = week_id, y = unit_id, fill = as.character(missing_outcome)), show.legend = FALSE)
    + theme(panel.background=element_rect(fill="#f5e7e7"),
            panel.border = element_blank())
    + scale_x_continuous(expand = c(0, 0))
    + scale_y_continuous(expand = c(0, 0))
    + scale_fill_manual(breaks = c("FALSE", "TRUE"), values = c("#7a0a0a", "#340505"))
    + labs(x = "Week", y = "Driver")
)
ggsave(file.path(OUTPUT_PATH, "panel_plot_drivers_times_missing_cohort_outcomes.pdf"), missing_cohort_outcome_panel_plot, width = 3.0625, height = 3.25)

# plot overlap graph

cohorts_expanded = panel_data |>
    select(rider_id, first_week_id, treatment_week_id) |>
    distinct() |>
    group_by(first_week_id, treatment_week_id) |>
    summarize(cohort_size = n()) |>
    ungroup() |>
    filter(cohort_size >= 100) |>
    arrange(desc(cohort_size)) |>
    mutate(cohort_id = row_number())

overlap_graph_nodes = cohorts_expanded |>
    select(cohort_id, cohort_size) |>
    distinct()

overlap_graph_edges = cohorts_expanded |>
    select(-c(cohort_size)) |>
    left_join(cohorts_expanded |> select(-c(cohort_size)), 
        by = join_by(overlaps(first_week_id, treatment_week_id, first_week_id, treatment_week_id, bounds = "[)")), 
        suffix = c("_l", "_r"), relationship = "many-to-many") |>
    filter(cohort_id_l != cohort_id_r) |>
    select(cohort_id_l, cohort_id_r, first_week_id_l, treatment_week_id_l, first_week_id_r, treatment_week_id_r) |>
    mutate(num_weeks_overlapping = (pmin(treatment_week_id_l, treatment_week_id_r) - pmax(first_week_id_l, first_week_id_r))) |>
    select(cohort_id_l, cohort_id_r, num_weeks_overlapping)

overlap_graph = graph_from_data_frame(d=overlap_graph_edges, vertices=overlap_graph_nodes, directed=FALSE)

overlap_graph_plot = (
    ggraph(overlap_graph)
    + theme_void()
    + geom_edge_link(aes(edge_width = num_weeks_overlapping), alpha = 0.2, color = "grey")
    + geom_node_point(aes(size = cohort_size), stroke = 0.25, shape = 21, fill = "#7a0a0a")
    + scale_edge_width_continuous(range = c(0.25, 1))
    + guides(size = "none", edge_width = "none")
)
ggsave(file.path(OUTPUT_PATH, "overlap_graph_1_plot_drivers_times.pdf"), overlap_graph_plot, width = 3.0625, height = 3.25)

library(Hmisc)
library(tidyverse)
library(arrow)

OUTPUT_PATH = Sys.getenv("OUTPUT_PATH")
leave_one_out_sim_path = file.path(OUTPUT_PATH, "leave_one_out_sims_provinces")
leave_one_out_sim_results_path = file.path(OUTPUT_PATH, "leave_one_out_sims_results_provinces")
dir.create(leave_one_out_sim_results_path)

boot_sim_ests = open_dataset(leave_one_out_sim_path)

err_stats = boot_sim_ests |>
    group_by(cohort_str, outcome, est_name) |>
    summarize(abs_bias = abs(mean(boot_outcome_mean - true_outcome_mean)),
              std_err = sd(boot_outcome_mean),
              rmse = sqrt(mean((boot_outcome_mean - true_outcome_mean)^2)),
              true_outcome_mean = max(true_outcome_mean),
              cohort_size = max(cohort_size)) |>
    collect() |>
    arrange(desc(cohort_size), cohort_str, outcome, est_name) |>
    pivot_longer(c(abs_bias, std_err, rmse), names_to = "err_stat")

err_stats_diff_twfe = err_stats |>
    filter(est_name != "twfe_clustered") |>
    inner_join(err_stats |>
        filter(est_name == "twfe_clustered") |>
        select(-est_name, -true_outcome_mean, -cohort_size),
        by = c("cohort_str", "outcome", "err_stat"), suffix = c("_other", "_twfe")) |>
    mutate(value_diff = value_twfe - value_other,
            value_ratio = value_other / value_twfe,
            cohort_str_clean = case_when(
                cohort_str == "orig_panel" ~ "Original Panel"
            ),
            err_stat_clean = factor(case_when(
                err_stat == "abs_bias" ~ "Absolute Bias",
                err_stat == "std_err" ~ "Standard Error",
                err_stat == "rmse" ~ "Root Mean Squared Error"
            ), levels = c("Absolute Bias", "Standard Error", "Root Mean Squared Error")),
            est_name_clean = case_when(
                est_name == "apm_rank_1" ~ "APM, Rank 1",
            )
        ) |>
    ungroup()

err_stat_summaries = err_stats_diff_twfe |>
    group_by(err_stat, est_name) |>
    summarize(
        n = n(),
        share_better_unweighted = mean(value_ratio < 1),
        share_better_weighted = sum(cohort_size * (value_ratio < 1)) / sum(cohort_size),
        mean_value_ratio_unweighted = mean(value_ratio),
        mean_value_ratio_weighted = sum(cohort_size * value_ratio) / sum(cohort_size),
        q25_value_ratio_unweighted = quantile(value_ratio, 0.25),
        q25_value_ratio_weighted = wtd.quantile(value_ratio, cohort_size, 0.25),
        q50_value_ratio_unweighted = quantile(value_ratio, 0.50),
        q50_value_ratio_weighted = wtd.quantile(value_ratio, cohort_size, 0.50),
        q75_value_ratio_unweighted = quantile(value_ratio, 0.75),
        q75_value_ratio_weighted = wtd.quantile(value_ratio, cohort_size, 0.75)
    )

plot_relative_error_metric_dist_for_panel_est = function(sim_summary_data_diff_twfe_for_panel_est, legend_pos = c(0.2, 0.84)) {
    rel_err_metric_ranks = sim_summary_data_diff_twfe_for_panel_est |>
        group_by(err_stat) |>
        arrange(value_ratio) |>
        mutate(rank = row_number(), cumu_cohort_size = cumsum(cohort_size)) |>
        ungroup()
    rel_err_metric_summaries = rel_err_metric_ranks |>
        group_by(err_stat) |>
        summarize(n = n(), 
            num_units = sum(cohort_size),
            share_value_ratios_below_one = mean(value_ratio < 1), 
            share_units_with_value_ratios_below_one = sum(cohort_size*(value_ratio < 1))/sum(cohort_size))
    rel_err_metric_dist = rel_err_metric_ranks |>
        inner_join(rel_err_metric_summaries, by = "err_stat") |>
        mutate(value_ratio_quantile = rank / n, value_ratio_units_quantile = cumu_cohort_size / num_units)

    (ggplot(rel_err_metric_dist)
        + theme_bw(base_size = 14)
        + geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 1)
        + geom_step(aes(y = value_ratio_units_quantile, x = value_ratio, color = err_stat), linewidth = 1.25)
        + geom_hline(aes(yintercept = share_units_with_value_ratios_below_one, color = err_stat), linetype = "dotted", linewidth = 1.25)
        + scale_x_continuous(trans="log10")
        + scale_y_continuous(limits = c(0, 1))
        + scale_color_viridis_d(limits = c("rmse", "std_err", "abs_bias"), 
            labels = c("Root Mean Squared Error", "Standard Error", "Absolute Bias"), end = 0.9)
        + labs(y = "Share w/ Lower Relative Error", x = "Relative Error: APM Error / TWFE Error", color = "Error Metric")
        + theme(legend.position = legend_pos,
            legend.background = element_rect(color="black", linewidth=0.5, linetype="solid")))
}

plot_error_metric_scatter_for_panel_est = function(sim_summary_data_diff_twfe_for_panel_est, relative_errs = FALSE) {
    plot_obj = ggplot(sim_summary_data_diff_twfe_for_panel_est, aes(x = value_twfe, y = value_other))
    labels_obj = labs(x = "TWFE Error Metric", y = "APM Error Metric")
    if (relative_errs) {
        plot_obj = ggplot(sim_summary_data_diff_twfe_for_panel_est,
        aes(x = value_twfe / true_outcome_mean, y = value_other / true_outcome_mean))
        labels_obj = labs(x = "TWFE Error Metric / True Outcome Mean", y = "APM Error Metric / True Outcome Mean")
    }
    (plot_obj
    + theme_bw(base_size = 14)
    + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 1.25)
    + geom_point(size = 3, alpha = 0.5)
    + scale_x_continuous(trans="log10")
    + scale_y_continuous(trans="log10")
    + facet_wrap(vars(err_stat_clean), scales = "free")
    + labels_obj)
}

for (curr_est_name in unique(err_stats_diff_twfe$est_name)) {
    rel_err_metrics_for_est = err_stats_diff_twfe |>
            filter(est_name == curr_est_name)
    ggsave(file.path(leave_one_out_sim_results_path, str_glue("rel_err_metrics_dists_{curr_est_name}.pdf")), 
        plot_relative_error_metric_dist_for_panel_est(rel_err_metrics_for_est), 
        width = 8, height = 4.5)
    ggsave(file.path(leave_one_out_sim_results_path, str_glue("abs_err_metrics_scatter_{curr_est_name}.pdf")), 
        plot_error_metric_scatter_for_panel_est(rel_err_metrics_for_est), 
        width = 8, height = 4.5)
    ggsave(file.path(leave_one_out_sim_results_path, str_glue("rel_err_metrics_scatter_{curr_est_name}.pdf")), 
        plot_error_metric_scatter_for_panel_est(rel_err_metrics_for_est, TRUE), 
        width = 8, height = 4.5)
}
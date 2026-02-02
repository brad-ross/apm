.validate_est_specs <- function(est_specs) {
  if (!is.list(est_specs) || length(est_specs) == 0L) stop("est_specs must be a non-empty list")
  for (i in seq_along(est_specs)) {
    sp <- est_specs[[i]]
    if (!is.list(sp)) stop(sprintf("est_specs[[%d]] must be a list", i))
    req <- c("factor_model_estimator", "include_outcome_fes", "r")
    miss <- setdiff(req, names(sp))
    if (length(miss) > 0L) stop(sprintf("spec %d missing fields: %s", i, paste(miss, collapse = ", ")))
  }
  invisible(TRUE)
}

.wrap_outcome_means_xptr_list <- function(res_named_xptr_list) {
  nms <- names(res_named_xptr_list)
  out <- setNames(vector("list", length(res_named_xptr_list)), nms)
  for (i in seq_along(res_named_xptr_list)) {
    out[[i]] <- OutcomeMeansEstimates$new(res_named_xptr_list[[i]])
  }
  out
}
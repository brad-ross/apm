#' Set APM Thread Count for Data Processing
#'
#' Sets the number of threads used by data.table operations within the apm
#' package. This affects panel processing and cohort construction.
#'
#' @description
#' This function configures parallelism for the R-level data.table operations
#' used in panel processing. For C++-level parallelism (used in estimation),
#' see the `num_threads` argument in functions like
#' \code{\link{est_cohort_specific_params}}.
#'
#' @param threads Integer; number of threads to use. If `NULL` (default),
#'   auto-detects and uses all logical cores on the system.
#'
#' @return Integer; the number of threads set (returned invisibly).
#'
#' @seealso \code{\link{get_cpp_default_concurrency}} for querying the C++
#'   default thread count.
#'
#' @examples
#' # Use all logical cores (default)
#' # set_apm_threads()
#'
#' # Use 4 threads
#' # set_apm_threads(4)
#'
#' # Use single-threaded mode
#' # set_apm_threads(1)
#'
#' @importFrom data.table setDTthreads
#' @importFrom parallel detectCores
#' @export
set_apm_threads <- function(threads = NULL) {
    if (is.null(threads)) {
        threads <- detectCores(logical = TRUE)
    }
    invisible(setDTthreads(threads))
}



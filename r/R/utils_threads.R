#' Set APM thread count
#'
#' Sets the number of threads used by data.table operations. If `threads` is
#' `NULL`, the number of logical cores on the system is used.
#'
#' @param threads Integer number of threads to use, or `NULL` to auto-detect.
#' @return Integer, the number of threads used (invisibly).
#' @examples
#' # Use all logical cores
#' # set_apm_threads()
#'
#' # Use 4 threads
#' # set_apm_threads(4)
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



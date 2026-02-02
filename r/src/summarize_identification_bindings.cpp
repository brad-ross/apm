#include <RcppArmadillo.h>
#include "r_utils.h"
#include "../../core/src/summarize_identification.h"

// [[Rcpp::depends(RcppArmadillo)]]

namespace {

inline Rcpp::List id_summary_to_r(const apm::IdentificationSummary& s) {
    return Rcpp::List::create(
        Rcpp::Named("largest_super_cohort_size") = static_cast<double>(s.largest_super_cohort_size),
        Rcpp::Named("largest_super_cohort_share") = s.largest_super_cohort_share,
        Rcpp::Named("min_cohort_size_in_largest_super") = static_cast<double>(s.min_cohort_size_in_largest_super),
        Rcpp::Named("num_outcomes_in_largest_super_cohort") = static_cast<double>(s.num_outcomes_in_largest_super_cohort),
        Rcpp::Named("total_outcome_weight_in_largest_super_cohort") = s.total_outcome_weight_in_largest_super_cohort,
        Rcpp::Named("share_outcomes_in_largest_super_cohort") = s.share_outcomes_in_largest_super_cohort,
        Rcpp::Named("share_outcome_weight_in_largest_super_cohort") = s.share_outcome_weight_in_largest_super_cohort,
        Rcpp::Named("num_o3_iterations") = static_cast<double>(s.num_o3_iterations)
    );
}

// Convert R-facing iteration selector to C++:
// - Positive k is 1-based in R and becomes (k-1) for C++ level indexing
// - Negative values (-1: final, -2: second-to-last, ...) are passed through
// - If NULL, return -1 (final iteration)
inline int r_iter_to_cpp(const Rcpp::Nullable<Rcpp::IntegerVector>& iter) {
    if (iter.isNull()) return -1;
    const int it = Rcpp::as<int>(iter.get());
    return (it > 0) ? (it - 1) : it;
}

} // anonymous namespace

//' Check if aligned factors are identified across all cohorts
//'
//' This function uses the Observed Outcome Overlap (O^3) algorithm to determine
//' if there is sufficient overlap in observed outcomes across all cohorts to
//' uniquely identify all factor vectors expressed with respect to a common basis.
//' Identification is achieved if the O^3 algorithm terminates with a single
//' super cohort containing all of the original cohorts.
//'
//' @param observed_outcome_indices A list of integer vectors, where each
//'   vector contains the 1-based indices for the observed outcomes for a cohort.
//' @param r The model rank (number of factors). This determines the minimum
//'   overlap required for cohorts to be merged in the O^3 algorithm.
//' @return A logical value: \code{TRUE} if the factors are identified across
//'   all cohorts, \code{FALSE} otherwise.
//' @details
//' Two cohorts can have their factors aligned if they share at least \code{r}
//' observed outcomes. The O^3 algorithm iteratively merges cohorts that meet
//' this criterion. If all cohorts eventually merge into a single super cohort,
//' the factors are considered identified.
//'
//' @seealso \code{\link{o3_algorithm}} for the full O^3 algorithm output,
//'   \code{\link{summarize_identification}} for comprehensive identification metrics.
//' @examples
//' # Three cohorts with outcomes that can be aligned
//' ooi <- list(c(1L, 2L, 3L), c(2L, 3L, 4L), c(3L, 4L, 5L))
//' aligned_factors_identified(ooi, r = 2L)  # TRUE
//'
//' # Cohorts without sufficient overlap
//' ooi_no_id <- list(c(1L, 2L), c(3L, 4L))
//' aligned_factors_identified(ooi_no_id, r = 2L)  # FALSE
//' @export
// [[Rcpp::export]]
bool aligned_factors_identified(
    Rcpp::List observed_outcome_indices,
    unsigned int r)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    return apm::aligned_factors_identified(ooi0, r);
}

//' Get the largest super cohort from O^3 algorithm
//'
//' Returns the indices of cohorts belonging to the largest super cohort at a
//' specified iteration of the O^3 (Observed Outcome Overlap) algorithm. The
//' largest super cohort is determined by the total number of units across its
//' member cohorts.
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of
//'   observed outcomes per cohort.
//' @param cohort_sizes An integer vector of cohort sizes (number of units),
//'   with the same length as \code{observed_outcome_indices}.
//' @param max_model_rank Maximum model rank (r). This determines the minimum
//'   overlap required for cohorts to be merged.
//' @param iter Optional iteration selector. Defaults to \code{NULL} (final
//'   iteration). Positive values are 1-based (e.g., 1 = first iteration).
//'   Negative values count from the end (-1 = final, -2 = second-to-last).
//' @return An integer vector of 1-based cohort indices belonging to the
//'   largest super cohort at the selected iteration.
//' @details
//' This function is useful for identifying which cohorts can have their
//' factor estimates aligned when full identification is not achieved. The
//' cohorts in the largest super cohort share sufficient outcome overlap to
//' allow factor alignment among themselves.
//'
//' @seealso \code{\link{o3_algorithm}} for the full algorithm trace,
//'   \code{\link{aligned_factors_identified}} for checking full identification,
//'   \code{\link{summarize_identification}} for comprehensive metrics.
//' @examples
//' ooi <- list(c(1L, 2L, 3L), c(2L, 3L, 4L), c(5L, 6L))
//' sizes <- c(100L, 150L, 50L)
//' # Get cohorts in largest super cohort at final iteration
//' get_largest_super_cohort(ooi, sizes, max_model_rank = 2L)
//' @export
// [[Rcpp::export]]
arma::uvec get_largest_super_cohort(
    Rcpp::List observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    Rcpp::Nullable<Rcpp::IntegerVector> iter = R_NilValue)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    const int it0 = r_iter_to_cpp(iter);
    arma::uvec res0 = apm::get_largest_super_cohort(ooi0, cohort_sizes, max_model_rank, it0);
    // Return 1-based indices to R
    return res0 + 1;
}

//' Summarize factor identification for a panel (C++ backend)
//'
//' Computes summary statistics about factor identification based on the O^3
//' (Observed Outcome Overlap) algorithm. This is the C++ implementation called
//' by the R wrapper \code{\link{summarize_identification}}.
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of
//'   observed outcomes per cohort.
//' @param cohort_sizes An integer vector of cohort sizes (number of units),
//'   with the same length as \code{observed_outcome_indices}.
//' @param max_model_rank Maximum model rank (r). This determines the minimum
//'   overlap required for cohorts to be merged.
//' @param iter Optional iteration selector. Defaults to \code{NULL} (final
//'   iteration). Positive values are 1-based. Negative values count from end.
//' @param outcome_weights Optional numeric vector of outcome weights. If
//'   provided, used to compute weighted outcome coverage statistics.
//' @return A named list with identification summary fields:
//'   \describe{
//'     \item{largest_super_cohort_size}{Number of cohorts in the largest super cohort}
//'     \item{largest_super_cohort_share}{Share of total units in largest super cohort}
//'     \item{min_cohort_size_in_largest_super}{Minimum cohort size within largest super cohort}
//'     \item{num_outcomes_in_largest_super_cohort}{Number of unique outcomes in largest super cohort}
//'     \item{total_outcome_weight_in_largest_super_cohort}{Total weight of outcomes in largest super cohort}
//'     \item{share_outcomes_in_largest_super_cohort}{Share of outcomes in largest super cohort}
//'     \item{share_outcome_weight_in_largest_super_cohort}{Share of outcome weight in largest super cohort}
//'     \item{num_o3_iterations}{Number of O^3 algorithm iterations}
//'   }
//' @seealso \code{\link{summarize_identification}} for the R wrapper,
//'   \code{\link{o3_algorithm}} for the full algorithm trace.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List summarize_identification_cpp(
    Rcpp::List observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    Rcpp::Nullable<Rcpp::IntegerVector> iter = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    apm::IdentificationSummary s;
    const int it0 = r_iter_to_cpp(iter);

    if (outcome_weights.isNull()) {
        s = apm::summarize_identification(ooi0, cohort_sizes, max_model_rank, it0);
    } else {
        arma::vec weights = Rcpp::as<arma::vec>(outcome_weights.get());
        s = apm::summarize_identification(ooi0, cohort_sizes, max_model_rank, it0, weights);
    }
    
    return id_summary_to_r(s);
}

//' Summarize factor identification for multiple panels (C++ backend)
//'
//' Computes identification summary statistics for multiple panels in a single
//' call. This is more efficient than calling \code{summarize_identification_cpp}
//' separately for each panel.
//'
//' @param observed_outcome_indices_list A list of lists, where each inner list
//'   contains integer vectors (1-based) of observed outcomes per cohort.
//' @param cohort_sizes_list A list of integer vectors of cohort sizes, one
//'   per panel. Each vector must match the length of the corresponding
//'   \code{observed_outcome_indices_list} element.
//' @param max_model_rank Maximum model rank (r). Applied to all panels.
//' @param iter Optional iteration selector. Defaults to \code{NULL} (final
//'   iteration). Applied to all panels.
//' @param outcome_weights_list Optional list of numeric vectors of outcome
//'   weights, one per panel. Defaults to equal weights for all panels.
//' @return A list of named lists, one per input panel, each with the same
//'   structure as \code{summarize_identification_cpp} output.
//' @seealso \code{\link{summarize_identification_cpp}} for single-panel version.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List summarize_identification_many_cpp(
    Rcpp::List observed_outcome_indices_list,
    Rcpp::List cohort_sizes_list,
    std::size_t max_model_rank,
    Rcpp::Nullable<Rcpp::IntegerVector> iter = R_NilValue,
    Rcpp::Nullable<Rcpp::List> outcome_weights_list = R_NilValue)
{
    if (observed_outcome_indices_list.size() != cohort_sizes_list.size()) {
        Rcpp::stop("observed_outcome_indices_list and cohort_sizes_list must have same length");
    }

    const int n = observed_outcome_indices_list.size();
    std::vector<apm::ObservedOutcomeIndices> all_ooi;
    std::vector<arma::uvec> all_sizes;
    all_ooi.reserve(n);
    all_sizes.reserve(n);

    for (int i = 0; i < n; ++i) {
        Rcpp::List ooi_i = observed_outcome_indices_list[i];
        all_ooi.push_back(apm::r_utils::to_cpp_observed_outcome_indices(ooi_i));
        arma::uvec sz_i = Rcpp::as<arma::uvec>(cohort_sizes_list[i]);
        all_sizes.push_back(std::move(sz_i));
    }

    std::vector<arma::vec> all_weights;
    bool has_weights = !outcome_weights_list.isNull();
    if (has_weights) {
        Rcpp::List weights_list(outcome_weights_list.get());
        if (weights_list.size() != n) {
            Rcpp::stop("outcome_weights_list must have same length as observed_outcome_indices_list");
        }
        all_weights.reserve(n);
        for (int i = 0; i < n; ++i) {
            Rcpp::NumericVector w_i = weights_list[i];
            all_weights.push_back(Rcpp::as<arma::vec>(w_i));
        }
    }

    std::vector<apm::IdentificationSummary> v;
    const int it0 = r_iter_to_cpp(iter);
    if (has_weights) {
        v = apm::summarize_identification(all_ooi, all_sizes, max_model_rank, it0, all_weights);
    } else {
        v = apm::summarize_identification(all_ooi, all_sizes, max_model_rank, it0);
    }
    Rcpp::List out(n);
    for (int i = 0; i < n; ++i) out[i] = id_summary_to_r(v[static_cast<std::size_t>(i)]);
    return out;
}

//' Count outcomes accessible via rank overlap for each cohort
//'
//' For each focal cohort, counts the (weighted) number of unique outcomes that
//' are observed in any cohort whose observed outcomes overlap with the focal
//' cohort in at least \code{rank} outcomes. The focal cohort's own outcomes
//' always contribute to its count. This measures how many outcomes each cohort
//' can potentially impute given the overlap structure.
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of
//'   observed outcomes per cohort.
//' @param rank Minimum number of overlapping observed outcomes required for
//'   another cohort to contribute its outcomes. Setting \code{rank = 0}
//'   includes all outcomes from all cohorts.
//' @param outcome_weights Optional numeric vector of outcome weights. If
//'   provided, must have length equal to the total number of outcomes.
//'   Defaults to equal weights (all ones).
//' @return A numeric vector of (weighted) counts, one per cohort. Each value
//'   represents the total weight of outcomes accessible to that cohort.
//' @details
//' This function helps assess how well each cohort can leverage information
//' from other cohorts for imputation. A cohort with a higher count has access
//' to more outcomes through sufficient overlap with other cohorts.
//'
//' When \code{rank} equals the model rank \code{r}, this indicates the number
//' of outcomes for which the cohort could potentially estimate factor loadings
//' and perform imputation.
//'
//' @seealso \code{\link{summarize_identification}} for comprehensive
//'   identification metrics, \code{\link{o3_algorithm}} for the overlap algorithm.
//' @examples
//' ooi <- list(c(1L, 2L, 3L), c(2L, 3L, 4L), c(5L, 6L))
//' # Count outcomes accessible with rank-2 overlap
//' count_outcomes_with_rank_overlap_per_cohort(ooi, rank = 2L)
//'
//' # With outcome weights
//' weights <- c(1, 2, 1, 1, 1, 1)  # outcome 2 weighted double
//' count_outcomes_with_rank_overlap_per_cohort(ooi, rank = 2L, outcome_weights = weights)
//' @export
// [[Rcpp::export]]
arma::vec count_outcomes_with_rank_overlap_per_cohort(
    Rcpp::List observed_outcome_indices,
    std::size_t rank,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    if (outcome_weights.isNull()) {
        return apm::count_outcomes_with_rank_overlap_per_cohort(ooi0, rank);
    }
    arma::vec weights = Rcpp::as<arma::vec>(outcome_weights.get());
    return apm::count_outcomes_with_rank_overlap_per_cohort(ooi0, rank, weights);
}

//' Mask observed outcome indices by cohort
//'
//' Applies a per-cohort outcome mask to a list of observed outcome indices,
//' removing specified outcomes from each cohort. This is useful for creating
//' hold-out sets for cross-validation or for simulating missing data patterns.
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of
//'   observed outcomes per cohort.
//' @param cohort_outcomes_to_mask_in Optional named list mapping 1-based cohort
//'   indices (as character names) to integer vectors of 1-based outcome indices
//'   to remove for that cohort. Names must be coercible to integers. If
//'   \code{NULL} or empty, the input is returned unchanged.
//' @return A list of integer vectors (1-based) with masked outcomes removed
//'   from each cohort's observed outcome set.
//' @details
//' This function is primarily used internally for computing error metrics
//' on held-out outcomes, but can also be useful for custom cross-validation
//' procedures.
//'
//' @examples
//' ooi <- list(c(1L, 2L, 3L), c(2L, 3L, 4L))
//' mask <- list(`1` = c(3L))  # Remove outcome 3 from cohort 1
//' get_masked_observed_outcome_indices(ooi, mask)
//' # Returns: list(c(1L, 2L), c(2L, 3L, 4L))
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List get_masked_observed_outcome_indices(
    Rcpp::List observed_outcome_indices,
    Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in = R_NilValue)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    apm::CohortOutcomeMask mask = apm::r_utils::to_cpp_mask(cohort_outcomes_to_mask_in);
    apm::ObservedOutcomeIndices masked = apm::get_masked_observed_outcome_indices(ooi0, mask);
    return apm::r_utils::to_r_observed_outcome_indices(masked);
}
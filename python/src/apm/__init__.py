"""Python interface to the APM matrix completion estimators."""

from ._core import (
    FactorModelEstimates,
    FactorModelEstimator,
    PCBase,
    PCEstimator,
    PCEstimatorWithFEs,
    WeightedBootstrap,
    aggregate_cohort_specific_covariate_coefs,
    aggregate_cohort_specific_outcome_fes,
    align_factors_using_apm,
    compute_aggregated_projection_matrix,
    get_version,
    get_weighted_bootstrap_draws,
    impute_outcomes_across_cohorts,
    impute_outcomes_across_cohorts_from_obs_outcomes,
    impute_outcomes_from_obs_outcomes,
)

__version__ = get_version()

__all__ = [
    "FactorModelEstimates",
    "FactorModelEstimator",
    "PCBase",
    "PCEstimator",
    "PCEstimatorWithFEs",
    "WeightedBootstrap",
    "aggregate_cohort_specific_covariate_coefs",
    "aggregate_cohort_specific_outcome_fes",
    "align_factors_using_apm",
    "compute_aggregated_projection_matrix",
    "get_version",
    "get_weighted_bootstrap_draws",
    "impute_outcomes_across_cohorts",
    "impute_outcomes_across_cohorts_from_obs_outcomes",
    "impute_outcomes_from_obs_outcomes",
]

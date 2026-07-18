"""High-level matrix completion and ATT estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl

from ._core import (
    PCEstimator,
    PCEstimatorWithFEs,
    aggregate_cohort_specific_outcome_fes,
    align_factors_using_apm,
    impute_outcomes_across_cohorts_from_obs_outcomes,
)
from .panel import BalancedPanel

COUNTERFACTUAL_COLUMN = "apm_counterfactual_mean"
EFFECT_COLUMN = "apm_att_contribution"


@dataclass(frozen=True)
class MatrixCompletionResult:
    """APM-completed cohort means and their long-panel representation.

    ``counterfactual_means`` is N x T, with each unit mapped to its completed
    cohort mean. It is suitable for ATT aggregation, not unit-level effect
    estimation.
    """

    panel: BalancedPanel
    factors: np.ndarray
    cohort_means: np.ndarray
    counterfactual_means: np.ndarray
    completed_panel: pl.DataFrame


@dataclass(frozen=True)
class ATTResult:
    """Average treatment effect on treated cells from APM completion.

    Each treated row's ATT contribution is its observed outcome minus its
    cohort's completed counterfactual mean. These contributions identify the
    aggregate ATT, not individual treatment effects.
    """

    att: float
    n_treated_cells: int
    att_by_outcome: pl.DataFrame
    completion: MatrixCompletionResult

    @property
    def completed_panel(self) -> pl.DataFrame:
        return self.completion.completed_panel

    @property
    def factors(self) -> np.ndarray:
        return self.completion.factors

    @property
    def cohort_means(self) -> np.ndarray:
        return self.completion.cohort_means

    @property
    def counterfactual_means(self) -> np.ndarray:
        return self.completion.counterfactual_means


def complete_panel(
    panel_df: Union[BalancedPanel, pl.DataFrame, pl.LazyFrame],
    *,
    unit_id_col: Optional[str] = None,
    outcome_id_col: Optional[str] = None,
    outcome_value_col: Optional[str] = None,
    treatment_col: Optional[str] = None,
    model_rank: Optional[int] = None,
    counterfactual_observed_col: Optional[str] = None,
    with_fixed_effects: bool = False,
) -> MatrixCompletionResult:
    """Reshape a balanced long panel and complete untreated cohort means."""

    panel = _coerce_panel(
        panel_df,
        unit_id_col=unit_id_col,
        outcome_id_col=outcome_id_col,
        outcome_value_col=outcome_value_col,
        treatment_col=treatment_col,
        model_rank=model_rank,
        counterfactual_observed_col=counterfactual_observed_col,
    )
    cohorts = panel.cohorts
    local_factors = []
    local_fixed_effects = []
    observed_means = []

    estimator_type = PCEstimatorWithFEs if with_fixed_effects else PCEstimator
    for units, indices in zip(
        cohorts.units, cohorts.observed_outcome_indices
    ):
        cohort_outcomes = panel.outcomes[np.ix_(units, indices)]
        estimates = (
            estimator_type(r=panel.model_rank, T_c=len(indices))
            .add_data(units, cohort_outcomes)
            .estimate()
        )
        local_factors.append(estimates.G())
        observed_means.append(cohort_outcomes.mean(axis=0))
        if with_fixed_effects:
            local_fixed_effects.append(estimates.g0())

    weights = cohorts.cohort_sizes.astype(float)
    try:
        factors = align_factors_using_apm(
            local_factors,
            cohorts.observed_outcome_indices,
            cohort_weights=weights,
        )
    except RuntimeError as error:
        raise ValueError(
            "cohort outcome overlap does not identify the requested factor model"
        ) from error

    fixed_effects = None
    if with_fixed_effects:
        fixed_effects = aggregate_cohort_specific_outcome_fes(
            local_fixed_effects,
            cohorts.observed_outcome_indices,
            cohort_weights=weights,
        )
    completed_means = impute_outcomes_across_cohorts_from_obs_outcomes(
        factors,
        cohorts.observed_outcome_indices,
        observed_means,
        g_0=fixed_effects,
    )
    for cohort_id, indices in enumerate(cohorts.observed_outcome_indices):
        completed_means[cohort_id, indices] = observed_means[cohort_id]

    counterfactual_means = completed_means[cohorts.unit_to_cohort]
    completed_panel = _make_completed_panel(panel, counterfactual_means)
    return MatrixCompletionResult(
        panel=panel,
        factors=factors,
        cohort_means=completed_means,
        counterfactual_means=counterfactual_means,
        completed_panel=completed_panel,
    )


def estimate_att(
    panel_df: Union[BalancedPanel, pl.DataFrame, pl.LazyFrame],
    *,
    unit_id_col: Optional[str] = None,
    outcome_id_col: Optional[str] = None,
    outcome_value_col: Optional[str] = None,
    treatment_col: Optional[str] = None,
    model_rank: Optional[int] = None,
    counterfactual_observed_col: Optional[str] = None,
    with_fixed_effects: bool = False,
) -> ATTResult:
    """Estimate ATT over treated cells in a balanced long Polars panel."""

    completion = complete_panel(
        panel_df,
        unit_id_col=unit_id_col,
        outcome_id_col=outcome_id_col,
        outcome_value_col=outcome_value_col,
        treatment_col=treatment_col,
        model_rank=model_rank,
        counterfactual_observed_col=counterfactual_observed_col,
        with_fixed_effects=with_fixed_effects,
    )
    panel = completion.panel
    treated = completion.completed_panel.filter(pl.col(panel.treatment_col))
    if treated.height == 0:
        raise ValueError("ATT requires at least one treated unit-outcome cell")

    att = float(treated.select(pl.col(EFFECT_COLUMN).mean()).item())
    att_by_outcome = (
        treated.group_by(panel.outcome_id_col)
        .agg(
            pl.col(EFFECT_COLUMN).mean().alias("att"),
            pl.col(EFFECT_COLUMN).len().alias("n_treated"),
        )
        .sort(panel.outcome_id_col)
    )
    return ATTResult(
        att=att,
        n_treated_cells=treated.height,
        att_by_outcome=att_by_outcome,
        completion=completion,
    )


def _coerce_panel(
    panel_df: Union[BalancedPanel, pl.DataFrame, pl.LazyFrame],
    *,
    unit_id_col: Optional[str],
    outcome_id_col: Optional[str],
    outcome_value_col: Optional[str],
    treatment_col: Optional[str],
    model_rank: Optional[int],
    counterfactual_observed_col: Optional[str],
) -> BalancedPanel:
    if isinstance(panel_df, BalancedPanel):
        supplied = {
            "unit_id_col": unit_id_col,
            "outcome_id_col": outcome_id_col,
            "outcome_value_col": outcome_value_col,
            "treatment_col": treatment_col,
            "model_rank": model_rank,
            "counterfactual_observed_col": counterfactual_observed_col,
        }
        conflicts = [name for name, value in supplied.items() if value is not None]
        if conflicts:
            raise ValueError(
                "column and rank arguments cannot be supplied with BalancedPanel: "
                + ", ".join(conflicts)
            )
        return panel_df

    columns = {
        "unit_id_col": unit_id_col,
        "outcome_id_col": outcome_id_col,
        "outcome_value_col": outcome_value_col,
        "treatment_col": treatment_col,
    }
    missing = [name for name, value in columns.items() if value is None]
    if missing:
        raise TypeError(
            "long panel input requires column arguments: " + ", ".join(missing)
        )
    return BalancedPanel(
        panel_df=panel_df,
        unit_id_col=unit_id_col,
        outcome_id_col=outcome_id_col,
        outcome_value_col=outcome_value_col,
        treatment_col=treatment_col,
        model_rank=1 if model_rank is None else model_rank,
        counterfactual_observed_col=counterfactual_observed_col,
    )


def _make_completed_panel(
    panel: BalancedPanel, counterfactual_means: np.ndarray
) -> pl.DataFrame:
    completed = panel.get_processed_panel().with_columns(
        pl.Series(COUNTERFACTUAL_COLUMN, counterfactual_means.reshape(-1))
    )
    return completed.with_columns(
        pl.when(pl.col(panel.treatment_col))
        .then(
            pl.col(panel.outcome_value_col) - pl.col(COUNTERFACTUAL_COLUMN)
        )
        .otherwise(None)
        .alias(EFFECT_COLUMN)
    )

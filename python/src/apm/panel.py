"""Polars-backed preparation for balanced causal panel data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

_UNIT_INDEX = "_apm_unit_index"
_OUTCOME_INDEX = "_apm_outcome_index"
_COHORT_ID = "_apm_cohort_id"
_FACTOR_OBSERVED = "_apm_factor_observed"
_RESULT_COLUMNS = {"apm_att_contribution", "apm_counterfactual_mean"}


@dataclass(frozen=True)
class CohortStructure:
    """Units grouped by their observed untreated outcome pattern."""

    units: Tuple[np.ndarray, ...]
    observed_outcome_indices: Tuple[np.ndarray, ...]
    cohort_sizes: np.ndarray
    unit_to_cohort: np.ndarray
    unit_cohorts: pl.DataFrame

    @property
    def num_cohorts(self) -> int:
        return len(self.units)


class BalancedPanel:
    """Validated long balanced panel prepared for APM estimation.

    The source data must contain exactly one row per unit-outcome pair. The
    treatment column marks cells whose untreated counterfactual is missing.
    An optional availability column can additionally exclude untreated cells
    from factor estimation without making the source panel itself unbalanced.
    """

    def __init__(
        self,
        panel_df: Union[pl.DataFrame, pl.LazyFrame],
        unit_id_col: str,
        outcome_id_col: str,
        outcome_value_col: str,
        treatment_col: str,
        model_rank: int = 1,
        counterfactual_observed_col: Optional[str] = None,
    ) -> None:
        self._unit_id_col = unit_id_col
        self._outcome_id_col = outcome_id_col
        self._outcome_value_col = outcome_value_col
        self._treatment_col = treatment_col
        self._counterfactual_observed_col = counterfactual_observed_col
        self._model_rank = _validate_model_rank(model_rank)

        frame = _collect_frame(panel_df)
        required = [unit_id_col, outcome_id_col, outcome_value_col, treatment_col]
        if counterfactual_observed_col is not None:
            required.append(counterfactual_observed_col)
        _validate_column_names(frame, required)
        self._original_panel = frame.clone()

        indexed, unit_ids, outcome_ids = _index_balanced_panel(
            frame, unit_id_col, outcome_id_col
        )
        indexed = _normalize_value_columns(
            indexed,
            outcome_value_col,
            treatment_col,
            counterfactual_observed_col,
        )
        _validate_absorbing_treatment(indexed, unit_id_col, treatment_col)

        self._unit_ids = unit_ids
        self._outcome_ids = outcome_ids
        self._num_units = len(unit_ids)
        self._num_outcomes = len(outcome_ids)
        shape = (self._num_units, self._num_outcomes)

        self._outcomes = (
            indexed.get_column(outcome_value_col).to_numpy().reshape(shape)
        )
        self._treatment = (
            indexed.get_column(treatment_col).to_numpy().reshape(shape)
        )
        if counterfactual_observed_col is None:
            available = np.ones(shape, dtype=bool)
        else:
            available = (
                indexed.get_column(counterfactual_observed_col)
                .to_numpy()
                .reshape(shape)
            )
        self._factor_observed = (~self._treatment) & available
        _validate_factor_observation_mask(self._factor_observed, self._model_rank)

        self._cohorts = _construct_cohorts(
            self._factor_observed, self._unit_ids, self._model_rank
        )
        cohort_ids = pl.DataFrame(
            {
                _UNIT_INDEX: np.arange(self._num_units, dtype=np.int64),
                _COHORT_ID: self._cohorts.unit_to_cohort,
            }
        )
        self._processed_panel = (
            indexed.join(cohort_ids, on=_UNIT_INDEX, how="left")
            .with_columns(
                pl.Series(_FACTOR_OBSERVED, self._factor_observed.reshape(-1))
            )
            .select(
                _UNIT_INDEX,
                _COHORT_ID,
                _OUTCOME_INDEX,
                *frame.columns,
                _FACTOR_OBSERVED,
            )
        )

    @property
    def model_rank(self) -> int:
        return self._model_rank

    @property
    def unit_ids(self) -> Sequence[Any]:
        return self._unit_ids

    @property
    def outcome_ids(self) -> Sequence[Any]:
        return self._outcome_ids

    @property
    def outcomes(self) -> np.ndarray:
        return self._outcomes

    @property
    def treatment(self) -> np.ndarray:
        return self._treatment

    @property
    def factor_observed(self) -> np.ndarray:
        return self._factor_observed

    @property
    def cohorts(self) -> CohortStructure:
        return self._cohorts

    @property
    def unit_id_col(self) -> str:
        return self._unit_id_col

    @property
    def outcome_id_col(self) -> str:
        return self._outcome_id_col

    @property
    def outcome_value_col(self) -> str:
        return self._outcome_value_col

    @property
    def treatment_col(self) -> str:
        return self._treatment_col

    def get_original_panel(self) -> pl.DataFrame:
        return self._original_panel.clone()

    def get_unit_id_col(self) -> str:
        return self._unit_id_col

    def get_outcome_id_col(self) -> str:
        return self._outcome_id_col

    def get_outcome_value_col(self) -> str:
        return self._outcome_value_col

    def get_treatment_col(self) -> str:
        return self._treatment_col

    def get_model_rank(self) -> int:
        return self._model_rank

    def get_processed_panel(self) -> pl.DataFrame:
        return self._processed_panel.clone()

    def get_unit_ids(self) -> Sequence[Any]:
        return list(self._unit_ids)

    def get_num_units(self) -> int:
        return self._num_units

    def get_outcome_ids(self) -> Sequence[Any]:
        return list(self._outcome_ids)

    def get_num_outcomes(self) -> int:
        return self._num_outcomes

    def get_outcome_to_index(self) -> dict:
        return {value: index for index, value in enumerate(self._outcome_ids)}

    def get_observed_outcome_indices(self) -> List[np.ndarray]:
        return [indices.copy() for indices in self._cohorts.observed_outcome_indices]

    def get_unit_cohorts(self) -> pl.DataFrame:
        return self._cohorts.unit_cohorts.clone()

    def get_num_cohorts(self) -> int:
        return self._cohorts.num_cohorts

    def get_cohort_sizes(self) -> np.ndarray:
        return self._cohorts.cohort_sizes.copy()

    def complete(self, with_fixed_effects: bool = False):
        """Complete cohort counterfactual means for this panel."""

        from .att import complete_panel

        return complete_panel(self, with_fixed_effects=with_fixed_effects)

    def estimate_att(self, with_fixed_effects: bool = False):
        """Estimate ATT for this panel."""

        from .att import estimate_att

        return estimate_att(self, with_fixed_effects=with_fixed_effects)


def construct_cohorts_from_panel(
    panel_df: Union[pl.DataFrame, pl.LazyFrame],
    unit_id_col: str,
    outcome_id_col: str,
    outcome_value_col: str,
    treatment_col: str,
    model_rank: int = 1,
    counterfactual_observed_col: Optional[str] = None,
) -> CohortStructure:
    """Construct counterfactual-observation cohorts from a long panel."""

    panel = BalancedPanel(
        panel_df=panel_df,
        unit_id_col=unit_id_col,
        outcome_id_col=outcome_id_col,
        outcome_value_col=outcome_value_col,
        treatment_col=treatment_col,
        model_rank=model_rank,
        counterfactual_observed_col=counterfactual_observed_col,
    )
    return panel.cohorts


def _collect_frame(
    panel_df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    if isinstance(panel_df, pl.LazyFrame):
        return panel_df.collect()
    if isinstance(panel_df, pl.DataFrame):
        return panel_df
    raise TypeError("panel_df must be a Polars DataFrame or LazyFrame")


def _validate_model_rank(model_rank: int) -> int:
    if isinstance(model_rank, bool) or not isinstance(model_rank, (int, np.integer)):
        raise TypeError("model_rank must be an integer")
    if model_rank < 1:
        raise ValueError("model_rank must be positive")
    return int(model_rank)


def _validate_column_names(frame: pl.DataFrame, required: Sequence[str]) -> None:
    if len(set(required)) != len(required):
        raise ValueError("panel column arguments must name distinct columns")
    reserved = {
        _UNIT_INDEX,
        _OUTCOME_INDEX,
        _COHORT_ID,
        _FACTOR_OBSERVED,
        *_RESULT_COLUMNS,
    }
    conflicts = reserved.intersection(frame.columns)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"panel contains reserved APM column(s): {names}")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"panel is missing required column(s): {', '.join(missing)}")
    if frame.height == 0:
        raise ValueError("panel must contain at least one row")
    null_columns = [
        name for name in required if frame.get_column(name).null_count() > 0
    ]
    if null_columns:
        raise ValueError(
            "panel columns cannot contain nulls: " + ", ".join(null_columns)
        )


def _index_balanced_panel(
    frame: pl.DataFrame, unit_id_col: str, outcome_id_col: str
) -> Tuple[pl.DataFrame, Sequence[Any], Sequence[Any]]:
    duplicate = frame.select(
        pl.struct(unit_id_col, outcome_id_col).is_duplicated().any()
    ).item()
    if duplicate:
        raise ValueError("panel must have exactly one row per unit-outcome pair")

    unit_index = (
        frame.select(unit_id_col)
        .unique()
        .sort(unit_id_col)
        .with_row_index(_UNIT_INDEX)
    )
    outcome_index = (
        frame.select(outcome_id_col)
        .unique()
        .sort(outcome_id_col)
        .with_row_index(_OUTCOME_INDEX)
    )
    expected_rows = unit_index.height * outcome_index.height
    if frame.height != expected_rows:
        raise ValueError(
            "panel must be balanced with every unit-outcome pair present; "
            f"expected {expected_rows} rows, found {frame.height}"
        )

    indexed = (
        frame.join(unit_index, on=unit_id_col, how="left")
        .join(outcome_index, on=outcome_id_col, how="left")
        .sort(_UNIT_INDEX, _OUTCOME_INDEX)
    )
    return (
        indexed,
        unit_index.get_column(unit_id_col).to_list(),
        outcome_index.get_column(outcome_id_col).to_list(),
    )


def _normalize_value_columns(
    frame: pl.DataFrame,
    outcome_value_col: str,
    treatment_col: str,
    counterfactual_observed_col: Optional[str],
) -> pl.DataFrame:
    treatment_values = set(frame.get_column(treatment_col).unique().to_list())
    if not treatment_values.issubset({0, 1, False, True}):
        raise ValueError("treatment must contain only boolean or 0/1 values")
    try:
        normalized = frame.with_columns(
            pl.col(outcome_value_col).cast(pl.Float64),
            pl.col(treatment_col).cast(pl.Int8),
        )
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as error:
        raise ValueError("outcome must be numeric and treatment must be 0/1") from error

    if not normalized.get_column(outcome_value_col).is_finite().all():
        raise ValueError("outcome values must be finite")

    expressions = [pl.col(treatment_col).cast(pl.Boolean)]
    if counterfactual_observed_col is not None:
        availability_values = set(
            frame.get_column(counterfactual_observed_col).unique().to_list()
        )
        if not availability_values.issubset({0, 1, False, True}):
            raise ValueError(
                "counterfactual availability must contain boolean or 0/1 values"
            )
        try:
            normalized.get_column(counterfactual_observed_col).cast(pl.Int8)
        except (
            pl.exceptions.InvalidOperationError,
            pl.exceptions.ComputeError,
        ) as error:
            raise ValueError(
                "counterfactual availability must contain boolean or 0/1 values"
            ) from error
        expressions.append(pl.col(counterfactual_observed_col).cast(pl.Boolean))
    return normalized.with_columns(*expressions)


def _validate_absorbing_treatment(
    frame: pl.DataFrame, unit_id_col: str, treatment_col: str
) -> None:
    reversals = frame.filter(
        pl.col(treatment_col)
        .cast(pl.Int8)
        .diff()
        .over(unit_id_col)
        .fill_null(0)
        < 0
    )
    if reversals.height:
        raise ValueError(
            "treatment must be absorbing within unit after one-shot adoption"
        )


def _validate_factor_observation_mask(mask: np.ndarray, model_rank: int) -> None:
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError(
            "each outcome period must have at least one untreated observed unit"
        )
    too_short = np.flatnonzero(mask.sum(axis=1) < model_rank)
    if len(too_short):
        raise ValueError(
            "every counterfactual-observation cohort must observe at least "
            f"model_rank={model_rank} outcomes"
        )


def _construct_cohorts(
    observed: np.ndarray, unit_ids: Sequence[Any], model_rank: int
) -> CohortStructure:
    patterns = {}
    for unit_index, row in enumerate(observed):
        pattern = tuple(np.flatnonzero(row).tolist())
        patterns.setdefault(pattern, []).append(unit_index)

    ordered = sorted(patterns.items(), key=lambda item: (len(item[0]), item[0]))
    units = tuple(
        np.asarray(unit_indices, dtype=np.int64) for _, unit_indices in ordered
    )
    outcome_indices = tuple(
        np.asarray(pattern, dtype=np.int64) for pattern, _ in ordered
    )
    if any(len(indices) < model_rank for indices in outcome_indices):
        raise ValueError(
            "every counterfactual-observation cohort must observe at least "
            f"model_rank={model_rank} outcomes"
        )

    unit_to_cohort = np.empty(len(unit_ids), dtype=np.int64)
    for cohort_id, cohort_units in enumerate(units):
        unit_to_cohort[cohort_units] = cohort_id
    cohort_sizes = np.asarray([len(values) for values in units], dtype=np.int64)
    unit_cohorts = pl.DataFrame(
        {
            "unit_id": list(unit_ids),
            "unit_index": np.arange(len(unit_ids), dtype=np.int64),
            "cohort_id": unit_to_cohort,
        }
    )
    return CohortStructure(
        units=units,
        observed_outcome_indices=outcome_indices,
        cohort_sizes=cohort_sizes,
        unit_to_cohort=unit_to_cohort,
        unit_cohorts=unit_cohorts,
    )

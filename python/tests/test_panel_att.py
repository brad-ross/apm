import numpy as np
import polars as pl
import pytest

import apm


def make_low_rank_panel(
    n_units=60,
    n_outcomes=16,
    n_treated=20,
    adoption=10,
    effect=2.0,
):
    rng = np.random.default_rng(42)
    factors = np.column_stack(
        (
            np.sin(np.linspace(0.0, 2.0 * np.pi, n_outcomes)),
            np.linspace(0.0, 1.0, n_outcomes),
        )
    )
    loadings = rng.normal(size=(n_units, 2))
    treated_units = np.argsort(loadings[:, 1])[-n_treated:]
    treatment = np.zeros((n_units, n_outcomes), dtype=bool)
    treatment[treated_units, adoption:] = True
    untreated = loadings @ factors.T
    outcomes = untreated + effect * treatment

    frame = pl.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), n_outcomes),
            "period": np.tile(np.arange(n_outcomes), n_units),
            "outcome": outcomes.reshape(-1),
            "treated": treatment.reshape(-1),
        }
    )
    return frame, untreated, treatment


def test_balanced_panel_reshapes_long_polars_data_and_exposes_metadata():
    frame, untreated, treatment = make_low_rank_panel()
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=9)

    panel = apm.BalancedPanel(
        shuffled,
        unit_id_col="unit",
        outcome_id_col="period",
        outcome_value_col="outcome",
        treatment_col="treated",
        model_rank=2,
    )

    assert panel.get_num_units() == untreated.shape[0]
    assert panel.get_num_outcomes() == untreated.shape[1]
    assert panel.get_model_rank() == 2
    assert panel.get_num_cohorts() == 2
    assert panel.get_unit_ids() == list(range(untreated.shape[0]))
    assert panel.get_outcome_ids() == list(range(untreated.shape[1]))
    assert panel.get_outcome_to_index() == {
        period: period for period in range(untreated.shape[1])
    }
    np.testing.assert_array_equal(panel.treatment, treatment)
    np.testing.assert_array_equal(
        np.sort(panel.get_cohort_sizes()), np.array([20, 40])
    )
    assert isinstance(panel.get_processed_panel(), pl.DataFrame)
    assert isinstance(panel.get_unit_cohorts(), pl.DataFrame)


def test_estimate_att_from_long_frame_recovers_rectangular_effect():
    frame, _, treatment = make_low_rank_panel(effect=1.75)
    frame = frame.with_columns((pl.col("unit") % 3).alias("auxiliary_group"))

    result = apm.estimate_att(
        frame.lazy(),
        unit_id_col="unit",
        outcome_id_col="period",
        outcome_value_col="outcome",
        treatment_col="treated",
        model_rank=2,
    )

    assert result.att == pytest.approx(1.75, abs=1e-10)
    assert result.n_treated_cells == int(treatment.sum())
    assert result.counterfactual_means.shape == treatment.shape
    assert result.cohort_means.shape == (2, treatment.shape[1])
    assert result.att_by_outcome.columns == ["period", "att", "n_treated"]
    assert "auxiliary_group" in result.completed_panel.columns
    assert result.completed_panel.filter(pl.col("treated")).select(
        pl.col("apm_att_contribution").is_not_null().all()
    ).item()
    assert result.completed_panel.filter(~pl.col("treated")).select(
        pl.col("apm_att_contribution").is_null().all()
    ).item()


def test_estimate_att_handles_staggered_and_noncontiguous_observation_masks():
    frame, untreated, _ = make_low_rank_panel(
        n_units=80,
        n_outcomes=18,
        n_treated=30,
        adoption=12,
        effect=1.5,
    )
    loading_proxy = untreated[:, -1] - untreated[:, 0]
    treated_units = np.argsort(loading_proxy)[-30:]
    treatment = np.zeros_like(untreated, dtype=bool)
    available = np.ones_like(treatment)
    for adoption, units, gaps in zip(
        (10, 13, 15),
        np.array_split(treated_units, 3),
        ((), (2, 5), (3, 7, 11)),
    ):
        treatment[units, adoption:] = True
        available[np.ix_(units, np.asarray(gaps, dtype=int))] = False

    outcomes = untreated + 1.5 * treatment
    jagged = pl.DataFrame(
        {
            "unit": np.repeat(np.arange(len(untreated)), untreated.shape[1]),
            "period": np.tile(np.arange(untreated.shape[1]), len(untreated)),
            "outcome": outcomes.reshape(-1),
            "treated": treatment.reshape(-1),
            "available": available.reshape(-1),
        }
    )
    panel = apm.BalancedPanel(
        jagged,
        unit_id_col="unit",
        outcome_id_col="period",
        outcome_value_col="outcome",
        treatment_col="treated",
        counterfactual_observed_col="available",
        model_rank=2,
    )
    result = panel.estimate_att()

    assert panel.get_num_cohorts() == 4
    assert result.att == pytest.approx(1.5, abs=1e-10)
    assert result.att_by_outcome.height == 8


def test_construct_cohorts_from_panel_matches_panel_object():
    frame, _, _ = make_low_rank_panel()
    cohorts = apm.construct_cohorts_from_panel(
        frame,
        unit_id_col="unit",
        outcome_id_col="period",
        outcome_value_col="outcome",
        treatment_col="treated",
        model_rank=2,
    )

    assert cohorts.num_cohorts == 2
    assert cohorts.unit_cohorts.columns == [
        "unit_id",
        "unit_index",
        "cohort_id",
    ]
    np.testing.assert_array_equal(np.sort(cohorts.cohort_sizes), [20, 40])


def test_complete_prepared_panel_supports_fixed_effects():
    frame, _, _ = make_low_rank_panel()
    panel = apm.BalancedPanel(
        frame,
        unit_id_col="unit",
        outcome_id_col="period",
        outcome_value_col="outcome",
        treatment_col="treated",
        model_rank=2,
    )

    completion = panel.complete(with_fixed_effects=True)

    assert completion.factors.shape == (16, 2)
    assert completion.cohort_means.shape == (2, 16)
    with pytest.raises(ValueError, match="cannot be supplied"):
        apm.complete_panel(panel, model_rank=2)


def test_panel_validation_rejects_malformed_designs():
    frame, _, _ = make_low_rank_panel()
    arguments = {
        "unit_id_col": "unit",
        "outcome_id_col": "period",
        "outcome_value_col": "outcome",
        "treatment_col": "treated",
        "model_rank": 2,
    }

    with pytest.raises(ValueError, match="balanced"):
        apm.BalancedPanel(frame.head(frame.height - 1), **arguments)
    with pytest.raises(ValueError, match="exactly one row"):
        apm.BalancedPanel(pl.concat([frame, frame.head(1)]), **arguments)

    nonabsorbing = frame.with_columns(
        pl.when((pl.col("unit") == 0) & (pl.col("period") == 1))
        .then(True)
        .when((pl.col("unit") == 0) & (pl.col("period") == 2))
        .then(False)
        .otherwise(pl.col("treated"))
        .alias("treated")
    )
    with pytest.raises(ValueError, match="absorbing"):
        apm.BalancedPanel(nonabsorbing, **arguments)

    fractional_treatment = frame.with_columns(
        pl.when((pl.col("unit") == 0) & (pl.col("period") == 0))
        .then(0.5)
        .otherwise(pl.col("treated").cast(pl.Float64))
        .alias("treated")
    )
    with pytest.raises(ValueError, match="0/1"):
        apm.BalancedPanel(fractional_treatment, **arguments)

    no_donors = frame.with_columns((pl.col("period") >= 10).alias("treated"))
    with pytest.raises(ValueError, match="at least one untreated"):
        apm.BalancedPanel(no_donors, **arguments)


def test_estimate_att_requires_treated_cells():
    frame, _, _ = make_low_rank_panel()
    untreated = frame.with_columns(pl.lit(False).alias("treated"))
    panel = apm.BalancedPanel(
        untreated,
        unit_id_col="unit",
        outcome_id_col="period",
        outcome_value_col="outcome",
        treatment_col="treated",
        model_rank=2,
    )

    assert panel.complete().cohort_means.shape == (1, 16)
    with pytest.raises(ValueError, match="at least one treated"):
        panel.estimate_att()

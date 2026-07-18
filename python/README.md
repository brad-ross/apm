# APM Python bindings

The Python package exposes the principal-components estimators and matrix
completion routines implemented by the APM C++ core.

## Install for development

From the repository root:

```bash
pixi run install-python-dev
pixi run test-python
```

## ATT from a long Polars panel

The high-level API accepts a balanced long panel with exactly one row per
unit-outcome pair. The treatment column must be an absorbing indicator that
switches on at adoption. APM reshapes the data, constructs cohorts, completes
the untreated cohort means, and aggregates the treated cells into an ATT. All
tabular validation, indexing, grouping, and result construction uses Polars;
NumPy arrays are used only at the C++ numerical boundary.

```python
import apm
import polars as pl

panel = pl.DataFrame(
    {
        "unit": [1, 1, 1, 2, 2, 2],
        "period": [1, 2, 3, 1, 2, 3],
        "outcome": [1.0, 2.0, 5.0, 2.0, 4.0, 6.0],
        "treated": [False, False, True, False, False, False],
    }
)

result = apm.estimate_att(
    panel,
    unit_id_col="unit",
    outcome_id_col="period",
    outcome_value_col="outcome",
    treatment_col="treated",
    model_rank=1,
)

result.att                 # scalar ATT over all treated cells
result.att_by_outcome      # Polars DataFrame of period-specific ATTs
result.completed_panel     # long Polars frame with completion/ATT contributions
result.counterfactual_means  # N x T NumPy matrix of cohort counterfactual means
```

For repeated estimation, prepare the panel once using the R-style container API:

```python
prepared = apm.BalancedPanel(
    panel,
    unit_id_col="unit",
    outcome_id_col="period",
    outcome_value_col="outcome",
    treatment_col="treated",
    model_rank=1,
)
result = prepared.estimate_att()
```

`BalancedPanel` exposes `get_unit_ids()`, `get_outcome_ids()`,
`get_observed_outcome_indices()`, `get_unit_cohorts()`, and the other panel
metadata accessors used by the R package. For balanced source data with
additional administrative gaps, pass a boolean `counterfactual_observed_col`;
the rows stay in the panel while unavailable cells are excluded from factor
estimation.

## Principal-components estimator

The estimator interface mirrors the R package. Python uses zero-based indices,
including unit indices, observed outcome indices, and bootstrap replicate
indices.

```python
import apm
import numpy as np

Y = np.array([
    [1.0, 2.0, 0.0],
    [2.0, 4.0, 0.0],
    [0.0, 0.0, 3.0],
])

estimator = apm.PCEstimator(r=1, T_c=3)
estimates = estimator.add_data(np.arange(len(Y)), Y).estimate()
G = estimates.G()
```

Use `PCEstimatorWithFEs` to center outcomes and estimate an outcome fixed-effect
vector, available through `estimates.g0()`.

## Matrix completion

Complete a cohort from its observed outcome means:

```python
G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
completed = apm.impute_outcomes_from_obs_outcomes(
    G,
    T_c=np.array([0, 1]),
    m_c=np.array([2.0, 3.0]),
)
# array([2., 3., 5.])
```

`align_factors_using_apm` combines cohort-specific factor estimates, while
`impute_outcomes_across_cohorts_from_obs_outcomes` completes all cohorts in one
call. Optional fixed effects and covariates follow the same `g_0`, `a`, and
`X_c` argument structure as the R API.

## Demo notebook

[`notebooks/matrix_completion_att.ipynb`](notebooks/matrix_completion_att.ipynb)
uses a latent-factor data-generating process to estimate ATT under both a
rectangular missing-outcome block and jagged, cohort-specific missingness. From
the repository root:

```bash
python -m pip install -e './python[demo]'
jupyter lab python/notebooks/matrix_completion_att.ipynb
```

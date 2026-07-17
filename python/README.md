# APM Python bindings

The Python package exposes the principal-components estimators and matrix
completion routines implemented by the APM C++ core.

## Install for development

From the repository root:

```bash
pixi run install-python-dev
pixi run test-python
```

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

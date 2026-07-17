import numpy as np
import pytest

import apm


def projection(matrix):
    q, _ = np.linalg.qr(matrix)
    return q @ q.T


def test_version_matches_package_version():
    assert apm.get_version() == "0.1.0"
    assert apm.__version__ == apm.get_version()


def test_pc_estimator_matches_principal_subspace_and_streams_batches():
    outcomes = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ]
    )
    estimator = apm.PCEstimator(r=1, T_c=2)

    returned = estimator.add_data([0, 1], outcomes[:2])
    returned.add_data(np.array([2, 3]), np.asfortranarray(outcomes[2:]))
    estimates = estimator.estimate()

    assert returned is estimator
    assert estimator.r() == 1
    assert estimator.T_c() == 2
    assert estimator.q() == 0
    assert estimator.B() == 0
    assert not estimates.has_bootstrap()
    assert not estimates.has_g0()
    assert estimates.g0() is None
    np.testing.assert_allclose(
        projection(estimates.G()),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        atol=1e-12,
    )


def test_pc_estimator_with_fixed_effects_matches_means():
    outcomes = np.array(
        [
            [1.0, 2.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    estimator = apm.PCEstimatorWithFEs(r=1, T_c=2)
    for index, row in enumerate(outcomes):
        estimator.add_datum(index, row)

    estimates = estimator.estimate()

    assert estimates.has_g0()
    np.testing.assert_allclose(estimates.g0(), [0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        projection(estimates.G()),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        atol=1e-12,
    )


@pytest.mark.parametrize("bootstrap_type", ["multinomial", "bayesian"])
def test_weighted_bootstrap_drives_estimator_replicates(bootstrap_type):
    outcomes = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ]
    )
    if bootstrap_type == "multinomial":
        bootstrap = apm.WeightedBootstrap(N=4, B=3, type=bootstrap_type, seed=42)
    else:
        bootstrap = apm.get_weighted_bootstrap_draws(
            N=4, B=3, type=bootstrap_type, seed=42
        )
    estimator = apm.PCEstimator(1, 2, bootstrap=bootstrap)
    estimator.add_data(np.arange(4), outcomes)
    estimates = estimator.estimate()

    assert bootstrap.n_obs() == 4
    assert bootstrap.n_bootstraps() == 3
    assert bootstrap.weights().shape == (4, 3)
    assert bootstrap.obs_rows([0, 3]).shape == (2, 3)
    np.testing.assert_allclose(bootstrap.weights().sum(axis=0), 1.0)
    assert estimator.B() == 3
    assert estimates.has_bootstrap()
    assert estimates.num_bootstraps() == 3
    assert estimates.G(0).shape == (2, 1)
    with pytest.raises(IndexError, match="bootstrap index"):
        estimates.G(3)


def test_bootstrap_none_seed_and_cohort_parameter_aggregation():
    bootstrap = apm.WeightedBootstrap(3, 2, seed=None)
    np.testing.assert_allclose(bootstrap.weights().sum(axis=0), 1.0)

    observed_indices = [[0, 1], [1, 2]]
    fixed_effects = [[10.0, 20.0], [30.0, 40.0]]
    np.testing.assert_allclose(
        apm.aggregate_cohort_specific_outcome_fes(
            fixed_effects, observed_indices, cohort_weights=[1.0, 1.0]
        ),
        [10.0, 25.0, 40.0],
    )
    np.testing.assert_allclose(
        apm.aggregate_cohort_specific_covariate_coefs(
            [[1.0, 3.0], [3.0, 1.0]], cohort_weights=[1.0, 3.0]
        ),
        [2.5, 1.5],
    )


def test_estimator_validates_shapes_and_indices():
    estimator = apm.PCEstimator(r=1, T_c=2)
    with pytest.raises(ValueError, match="Y must be N x T_c"):
        estimator.add_data([0, 1], np.ones((2, 3)))
    with pytest.raises(ValueError, match="negative"):
        estimator.add_data([0, -1], np.ones((2, 2)))
    with pytest.raises(ValueError, match="X must be empty"):
        estimator.add_data([0, 1], np.ones((2, 2)), np.ones((2, 2, 1)))

    estimator_with_covariates = apm.PCEstimator(r=1, T_c=2, q=1)
    estimator_with_covariates.add_data(
        [0, 1], np.ones((2, 2)), np.ones((2, 2, 1))
    )
    assert estimator_with_covariates.q() == 1


def test_single_cohort_matrix_completion_with_optional_components():
    factors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    observed_indices = np.array([0, 1])
    loadings = np.array([2.0, 3.0])
    expected = factors @ loadings

    completed = apm.impute_outcomes_from_obs_outcomes(
        factors[:, ::-1][:, ::-1], observed_indices, expected[observed_indices]
    )
    np.testing.assert_allclose(completed, expected, atol=1e-12)

    g_0 = np.array([10.0, 20.0, 30.0])
    completed_with_fes = apm.impute_outcomes_from_obs_outcomes(
        factors,
        observed_indices,
        (expected + g_0)[observed_indices],
        g_0=g_0,
    )
    np.testing.assert_allclose(completed_with_fes, expected + g_0, atol=1e-12)

    a = np.array([0.5, -1.0])
    X_c = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]])
    expected_with_all_components = expected + g_0 + X_c @ a
    completed_with_all_components = apm.impute_outcomes_from_obs_outcomes(
        factors,
        observed_indices,
        expected_with_all_components[observed_indices],
        g_0=g_0,
        a=a,
        X_c=X_c,
    )
    np.testing.assert_allclose(
        completed_with_all_components, expected_with_all_components, atol=1e-12
    )

    with pytest.raises(ValueError, match="a and X_c"):
        apm.impute_outcomes_from_obs_outcomes(
            factors, observed_indices, expected[observed_indices], a=[1.0]
        )
    with pytest.raises(IndexError, match="outside"):
        apm.impute_outcomes_from_obs_outcomes(factors, [0, 3], [2.0, 3.0])
    with pytest.raises(ValueError, match="m_c length"):
        apm.impute_outcomes_from_obs_outcomes(factors, [0, 1], [2.0])


def test_multi_cohort_completion_from_observed_outcomes_and_loadings():
    factors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    loadings = np.array([[2.0, 3.0], [-1.0, 4.0]])
    expected = loadings @ factors.T
    observed_indices = [np.array([0, 1]), np.array([1, 2])]
    observed_outcomes = [
        expected[0, observed_indices[0]],
        expected[1, observed_indices[1]],
    ]

    from_observed = apm.impute_outcomes_across_cohorts_from_obs_outcomes(
        factors, observed_indices, observed_outcomes
    )
    from_loadings = apm.impute_outcomes_across_cohorts(factors, loadings)

    np.testing.assert_allclose(from_observed, expected, atol=1e-12)
    np.testing.assert_allclose(from_loadings, expected, atol=1e-12)


def test_apm_alignment_recovers_global_factor_subspace():
    factors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    observed_indices = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    transforms = [
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[2.0, 0.0], [0.0, 0.5]]),
    ]
    cohort_factors = [
        factors[indices] @ transform
        for indices, transform in zip(observed_indices, transforms)
    ]

    aligned = apm.align_factors_using_apm(
        cohort_factors, observed_indices, cohort_weights=[1.0, 2.0]
    )
    apm_matrix = apm.compute_aggregated_projection_matrix(
        cohort_factors, observed_indices
    )

    assert aligned.shape == factors.shape
    assert apm_matrix.shape == (4, 4)
    np.testing.assert_allclose(projection(aligned), projection(factors), atol=1e-10)


def test_end_to_end_estimate_align_and_complete():
    true_factors = np.array([[1.0], [2.0], [3.0], [4.0]])
    observed_indices = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    unit_loadings = [np.array([1.0, 2.0, -1.0]), np.array([0.5, -2.0, 3.0])]

    local_factors = []
    for indices, loadings in zip(observed_indices, unit_loadings):
        outcomes = loadings[:, None] @ true_factors[indices].T
        estimator = apm.PCEstimator(r=1, T_c=len(indices))
        local_factors.append(
            estimator.add_data(np.arange(len(loadings)), outcomes).estimate().G()
        )

    factors = apm.align_factors_using_apm(local_factors, observed_indices)
    cohort_loadings = np.array([1.5, -0.75])
    expected = cohort_loadings[:, None] * true_factors.T
    observed_outcomes = [
        expected[cohort, indices]
        for cohort, indices in enumerate(observed_indices)
    ]

    completed = apm.impute_outcomes_across_cohorts_from_obs_outcomes(
        factors, observed_indices, observed_outcomes
    )
    np.testing.assert_allclose(completed, expected, atol=1e-10)

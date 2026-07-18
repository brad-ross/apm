#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "apm_core.h"
#include "bootstrap.h"
#include "cohort_specific_param_structs.h"
#include "factor_model_estimators/FactorModelEstimator.h"
#include "factor_model_estimators/pc_estimators.h"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<py::ssize_t, py::array::c_style | py::array::forcecast>;

DoubleArray require_double_array(const py::handle& value, int ndim, const char* name) {
    DoubleArray array = DoubleArray::ensure(value);
    if (!array) {
        throw py::type_error(std::string(name) + " must be convertible to a NumPy float64 array");
    }
    if (array.ndim() != ndim) {
        throw py::value_error(
            std::string(name) + " must be " + std::to_string(ndim) + "-dimensional");
    }
    return array;
}

arma::mat to_mat(const py::handle& value, const char* name) {
    DoubleArray array = require_double_array(value, 2, name);
    const auto input = array.unchecked<2>();
    arma::mat output(
        static_cast<arma::uword>(input.shape(0)),
        static_cast<arma::uword>(input.shape(1)));

    for (py::ssize_t row = 0; row < input.shape(0); ++row) {
        for (py::ssize_t col = 0; col < input.shape(1); ++col) {
            output(static_cast<arma::uword>(row), static_cast<arma::uword>(col)) =
                input(row, col);
        }
    }
    return output;
}

arma::vec to_vec(const py::handle& value, const char* name) {
    DoubleArray array = require_double_array(value, 1, name);
    const auto input = array.unchecked<1>();
    arma::vec output(static_cast<arma::uword>(input.shape(0)));

    for (py::ssize_t i = 0; i < input.shape(0); ++i) {
        output(static_cast<arma::uword>(i)) = input(i);
    }
    return output;
}

arma::cube to_cube(const py::handle& value, const char* name) {
    DoubleArray array = require_double_array(value, 3, name);
    const auto input = array.unchecked<3>();
    arma::cube output(
        static_cast<arma::uword>(input.shape(0)),
        static_cast<arma::uword>(input.shape(1)),
        static_cast<arma::uword>(input.shape(2)));

    for (py::ssize_t row = 0; row < input.shape(0); ++row) {
        for (py::ssize_t col = 0; col < input.shape(1); ++col) {
            for (py::ssize_t slice = 0; slice < input.shape(2); ++slice) {
                output(
                    static_cast<arma::uword>(row),
                    static_cast<arma::uword>(col),
                    static_cast<arma::uword>(slice)) = input(row, col, slice);
            }
        }
    }
    return output;
}

arma::uvec to_uvec(const py::handle& value, const char* name) {
    IndexArray array = IndexArray::ensure(value);
    if (!array) {
        throw py::type_error(std::string(name) + " must be convertible to an integer array");
    }
    if (array.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be 1-dimensional");
    }

    const auto input = array.unchecked<1>();
    arma::uvec output(static_cast<arma::uword>(input.shape(0)));
    for (py::ssize_t i = 0; i < input.shape(0); ++i) {
        if (input(i) < 0) {
            throw py::value_error(std::string(name) + " cannot contain negative indices");
        }
        output(static_cast<arma::uword>(i)) = static_cast<arma::uword>(input(i));
    }
    return output;
}

void validate_indices_within(
    const arma::uvec& indices,
    arma::uword size,
    const char* name) {
    if (!indices.empty() && indices.max() >= size) {
        throw py::index_error(
            std::string(name) + " contains an index outside the outcome matrix");
    }
}

py::array_t<double> to_numpy(const arma::mat& input) {
    py::array_t<double> output({
        static_cast<py::ssize_t>(input.n_rows),
        static_cast<py::ssize_t>(input.n_cols)});
    auto result = output.mutable_unchecked<2>();
    for (arma::uword row = 0; row < input.n_rows; ++row) {
        for (arma::uword col = 0; col < input.n_cols; ++col) {
            result(static_cast<py::ssize_t>(row), static_cast<py::ssize_t>(col)) =
                input(row, col);
        }
    }
    return output;
}

py::array_t<double> to_numpy(const arma::vec& input) {
    py::array_t<double> output(static_cast<py::ssize_t>(input.n_elem));
    auto result = output.mutable_unchecked<1>();
    for (arma::uword i = 0; i < input.n_elem; ++i) {
        result(static_cast<py::ssize_t>(i)) = input(i);
    }
    return output;
}

std::vector<arma::mat> to_mats(const py::sequence& values, const char* name) {
    std::vector<arma::mat> output;
    output.reserve(static_cast<std::size_t>(py::len(values)));
    for (py::handle value : values) {
        output.push_back(to_mat(value, name));
    }
    return output;
}

std::vector<arma::vec> to_vecs(const py::sequence& values, const char* name) {
    std::vector<arma::vec> output;
    output.reserve(static_cast<std::size_t>(py::len(values)));
    for (py::handle value : values) {
        output.push_back(to_vec(value, name));
    }
    return output;
}

apm::ObservedOutcomeIndices to_observed_indices(const py::sequence& values) {
    apm::ObservedOutcomeIndices output;
    output.reserve(static_cast<std::size_t>(py::len(values)));
    for (py::handle value : values) {
        output.push_back(to_uvec(value, "observed_outcome_indices element"));
    }
    return output;
}

std::optional<arma::vec> optional_vec(const py::object& value, const char* name) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return to_vec(value, name);
}

std::optional<arma::mat> optional_mat(const py::object& value, const char* name) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return to_mat(value, name);
}

const apm::FactorModelParameters& select_parameters(
    const apm::FactorModelEstimates& estimates,
    const py::object& bootstrap_index) {
    if (bootstrap_index.is_none()) {
        return estimates.parameter_estimates;
    }

    const py::ssize_t index = py::cast<py::ssize_t>(bootstrap_index);
    if (index < 0 || static_cast<std::size_t>(index) >= estimates.bootstrap_replicates.size()) {
        throw py::index_error("bootstrap index out of range");
    }
    return estimates.bootstrap_replicates[static_cast<std::size_t>(index)];
}

std::shared_ptr<apm::WeightedBootstrap> make_bootstrap(
    std::size_t n,
    std::size_t b,
    std::string type,
    std::uint64_t seed) {
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (type == "multinomial") {
        return std::make_shared<apm::MultinomialBootstrap>(n, b, seed);
    }
    if (type == "bayesian") {
        return std::make_shared<apm::BayesianBootstrap>(n, b, seed);
    }
    throw py::value_error("type must be either 'multinomial' or 'bayesian'");
}

std::uint64_t optional_seed(const py::object& seed) {
    return seed.is_none() ? 0 : py::cast<std::uint64_t>(seed);
}

std::shared_ptr<const apm::WeightedBootstrap> optional_bootstrap(
    const py::object& bootstrap) {
    if (bootstrap.is_none()) {
        return nullptr;
    }
    return py::cast<std::shared_ptr<apm::WeightedBootstrap>>(bootstrap);
}

arma::vec impute_from_observed(
    const arma::mat& G,
    const arma::uvec& observed_indices,
    const arma::vec& observed_outcomes,
    const std::optional<arma::vec>& g_0,
    const std::optional<arma::vec>& a,
    const std::optional<arma::mat>& X_c) {
    validate_indices_within(observed_indices, G.n_rows, "T_c");
    if (observed_indices.n_elem != observed_outcomes.n_elem) {
        throw py::value_error("m_c length must match T_c length");
    }
    if (g_0 && g_0->n_elem != G.n_rows) {
        throw py::value_error("g_0 length must match the number of rows in G");
    }
    if (a.has_value() != X_c.has_value()) {
        throw py::value_error("a and X_c must be provided together");
    }
    if (a && (X_c->n_rows != G.n_rows || X_c->n_cols != a->n_elem)) {
        throw py::value_error("X_c must have shape (G rows, a length)");
    }
    if (g_0 && a) {
        return apm::impute_outcomes_from_obs_outcomes(
            G, *g_0, *a, observed_indices, observed_outcomes, *X_c);
    }
    if (g_0) {
        return apm::impute_outcomes_from_obs_outcomes(
            G, *g_0, observed_indices, observed_outcomes);
    }
    if (a) {
        return apm::impute_outcomes_from_obs_outcomes(
            G, *a, observed_indices, observed_outcomes, *X_c);
    }
    return apm::impute_outcomes_from_obs_outcomes(G, observed_indices, observed_outcomes);
}

arma::mat impute_all_from_observed(
    const arma::mat& G,
    const apm::ObservedOutcomeIndices& observed_indices,
    const std::vector<arma::vec>& observed_outcomes,
    const std::optional<arma::vec>& g_0,
    const std::optional<arma::vec>& a,
    const std::optional<std::vector<arma::mat>>& X_c) {
    if (observed_indices.size() != observed_outcomes.size()) {
        throw py::value_error(
            "observed_outcome_indices and m_c_vec must have the same length");
    }
    for (std::size_t cohort = 0; cohort < observed_indices.size(); ++cohort) {
        validate_indices_within(
            observed_indices[cohort], G.n_rows, "observed_outcome_indices element");
        if (observed_indices[cohort].n_elem != observed_outcomes[cohort].n_elem) {
            throw py::value_error(
                "each m_c_vec length must match its observed outcome index length");
        }
    }
    if (g_0 && g_0->n_elem != G.n_rows) {
        throw py::value_error("g_0 length must match the number of rows in G");
    }
    if (a.has_value() != X_c.has_value()) {
        throw py::value_error("a and X_c_vec must be provided together");
    }
    if (a) {
        if (X_c->size() != observed_indices.size()) {
            throw py::value_error("X_c_vec must contain one matrix per cohort");
        }
        for (const arma::mat& X : *X_c) {
            if (X.n_rows != G.n_rows || X.n_cols != a->n_elem) {
                throw py::value_error(
                    "each X_c_vec matrix must have shape (G rows, a length)");
            }
        }
    }
    if (g_0 && a) {
        return apm::impute_outcomes_across_cohorts_from_obs_outcomes(
            G, *g_0, *a, observed_indices, observed_outcomes, *X_c);
    }
    if (g_0) {
        return apm::impute_outcomes_across_cohorts_from_obs_outcomes(
            G, *g_0, observed_indices, observed_outcomes);
    }
    if (a) {
        return apm::impute_outcomes_across_cohorts_from_obs_outcomes(
            G, *a, observed_indices, observed_outcomes, *X_c);
    }
    return apm::impute_outcomes_across_cohorts_from_obs_outcomes(
        G, observed_indices, observed_outcomes);
}

arma::mat impute_all_from_loadings(
    const arma::mat& G,
    const arma::mat& L,
    const std::optional<arma::vec>& g_0,
    const std::optional<arma::vec>& a,
    const std::optional<std::vector<arma::mat>>& X_c) {
    if (G.n_cols != L.n_cols) {
        throw py::value_error("G and L must have the same number of columns");
    }
    if (g_0 && g_0->n_elem != G.n_rows) {
        throw py::value_error("g_0 length must match the number of rows in G");
    }
    if (a.has_value() != X_c.has_value()) {
        throw py::value_error("a and X_c must be provided together");
    }
    if (a) {
        if (X_c->size() != L.n_rows) {
            throw py::value_error("X_c must contain one matrix per row of L");
        }
        for (const arma::mat& X : *X_c) {
            if (X.n_rows != G.n_rows || X.n_cols != a->n_elem) {
                throw py::value_error(
                    "each X_c matrix must have shape (G rows, a length)");
            }
        }
    }
    if (g_0 && a) {
        return apm::impute_outcomes_across_cohorts(G, *g_0, *a, *X_c, L);
    }
    if (g_0) {
        return apm::impute_outcomes_across_cohorts(G, *g_0, L);
    }
    if (a) {
        return apm::impute_outcomes_across_cohorts(G, *a, *X_c, L);
    }
    return apm::impute_outcomes_across_cohorts(G, L);
}

void bind_estimator_methods(py::class_<
                            apm::FactorModelEstimator,
                            std::shared_ptr<apm::FactorModelEstimator>>& cls) {
    cls.def(
           "add_data",
           [](apm::FactorModelEstimator& self,
              const py::handle& unit_indices,
              const py::handle& outcomes,
              const py::object& covariates) -> apm::FactorModelEstimator& {
               arma::uvec indices = to_uvec(unit_indices, "unit_idxs");
               arma::mat Y = to_mat(outcomes, "Y");
               arma::cube X = covariates.is_none()
                   ? arma::cube()
                   : to_cube(covariates, "X");
               {
                   py::gil_scoped_release release;
                   self.add_data(indices, Y, X);
               }
               return self;
           },
           py::arg("unit_idxs"),
           py::arg("Y"),
           py::arg("X") = py::none(),
           py::return_value_policy::reference_internal,
           "Add an N x T_c outcome batch and optional N x T_c x q covariates.")
        .def(
            "add_datum",
            [](apm::FactorModelEstimator& self,
               std::size_t unit_index,
               const py::handle& outcomes,
               const py::object& covariates) -> apm::FactorModelEstimator& {
                arma::vec Y = to_vec(outcomes, "Y");
                arma::mat X = covariates.is_none()
                    ? arma::mat()
                    : to_mat(covariates, "X");
                {
                    py::gil_scoped_release release;
                    self.add_datum(unit_index, Y, X);
                }
                return self;
            },
            py::arg("unit_idx"),
            py::arg("Y"),
            py::arg("X") = py::none(),
            py::return_value_policy::reference_internal,
            "Add one unit's outcomes and optional T_c x q covariates.")
        .def(
            "estimate",
            &apm::FactorModelEstimator::estimate,
            py::call_guard<py::gil_scoped_release>(),
            "Finalize estimation and return factor model estimates.")
        .def("r", &apm::FactorModelEstimator::r)
        .def("T_c", &apm::FactorModelEstimator::T_c)
        .def("q", &apm::FactorModelEstimator::q)
        .def("B", &apm::FactorModelEstimator::num_bootstraps)
        .def("has_bootstrap", &apm::FactorModelEstimator::has_bootstrap);
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "Python bindings for the APM C++ matrix completion estimators";

    module.def("get_version", &apm::get_version, "Return the APM core version.");

    py::class_<apm::WeightedBootstrap, std::shared_ptr<apm::WeightedBootstrap>>(
        module,
        "WeightedBootstrap",
        "Immutable normalized bootstrap weights with shape N x B.")
        .def(
            py::init([](std::size_t n,
                        std::size_t b,
                        const std::string& type,
                        const py::object& seed) {
                return make_bootstrap(n, b, type, optional_seed(seed));
            }),
            py::arg("N"),
            py::arg("B"),
            py::arg("type") = "multinomial",
            py::arg("seed") = py::none())
        .def("n_obs", &apm::WeightedBootstrap::n_obs)
        .def("n_bootstraps", &apm::WeightedBootstrap::n_bootstraps)
        .def(
            "draw",
            [](const apm::WeightedBootstrap& self, std::size_t b) {
                return to_numpy(self.draw(b));
            },
            py::arg("b"))
        .def(
            "obs",
            [](const apm::WeightedBootstrap& self, std::size_t i) {
                return to_numpy(self.obs(i));
            },
            py::arg("i"))
        .def(
            "obs_rows",
            [](const apm::WeightedBootstrap& self, const py::handle& indices) {
                return to_numpy(self.obs(to_uvec(indices, "idx")));
            },
            py::arg("idx"))
        .def("weights", [](const apm::WeightedBootstrap& self) {
            return to_numpy(self.weights());
        })
        .def("desc", &apm::WeightedBootstrap::desc)
        .def("__repr__", [](const apm::WeightedBootstrap& self) {
            return "<apm.WeightedBootstrap " + self.desc() + ">";
        });

    module.def(
        "get_weighted_bootstrap_draws",
        [](std::size_t n,
           std::size_t b,
           const std::string& type,
           const py::object& seed) {
            return make_bootstrap(n, b, type, optional_seed(seed));
        },
        py::arg("N"),
        py::arg("B"),
        py::arg("type") = "multinomial",
        py::arg("seed") = py::none(),
        "Construct normalized N x B bootstrap weights.");

    py::class_<apm::FactorModelEstimates>(
        module,
        "FactorModelEstimates",
        "Point factor model estimates and optional bootstrap replicates.")
        .def("has_bootstrap", &apm::FactorModelEstimates::has_bootstrap_replicates)
        .def("num_bootstraps", &apm::FactorModelEstimates::n_bootstrap_replicates)
        .def("has_g0", [](const apm::FactorModelEstimates& self) {
            return self.parameter_estimates.g_0.has_value();
        })
        .def("has_a", [](const apm::FactorModelEstimates& self) {
            return self.parameter_estimates.a.has_value();
        })
        .def("has_L", [](const apm::FactorModelEstimates& self) {
            return self.parameter_estimates.L.has_value();
        })
        .def(
            "G",
            [](const apm::FactorModelEstimates& self, const py::object& b) {
                return to_numpy(select_parameters(self, b).G);
            },
            py::arg("b") = py::none())
        .def(
            "g0",
            [](const apm::FactorModelEstimates& self, const py::object& b) -> py::object {
                const auto& value = select_parameters(self, b).g_0;
                if (!value) {
                    return py::none();
                }
                return to_numpy(*value);
            },
            py::arg("b") = py::none())
        .def(
            "a",
            [](const apm::FactorModelEstimates& self, const py::object& b) -> py::object {
                const auto& value = select_parameters(self, b).a;
                if (!value) {
                    return py::none();
                }
                return to_numpy(*value);
            },
            py::arg("b") = py::none())
        .def(
            "L",
            [](const apm::FactorModelEstimates& self, const py::object& b) -> py::object {
                const auto& value = select_parameters(self, b).L;
                if (!value) {
                    return py::none();
                }
                return to_numpy(*value);
            },
            py::arg("b") = py::none());

    auto estimator = py::class_<
        apm::FactorModelEstimator,
        std::shared_ptr<apm::FactorModelEstimator>>(
        module, "FactorModelEstimator", "Abstract streaming estimator interface.");
    bind_estimator_methods(estimator);

    py::class_<apm::PCBase,
               apm::FactorModelEstimator,
               std::shared_ptr<apm::PCBase>>(
        module, "PCBase", "Abstract base for principal-components estimators.");

    py::class_<apm::PCEstimator,
               apm::PCBase,
               std::shared_ptr<apm::PCEstimator>>(
        module,
        "PCEstimator",
        "Streaming principal-components factor estimator.")
        .def(
            py::init([](std::size_t r,
                        std::size_t T_c,
                        const py::object& bootstrap,
                        std::size_t q) {
                return std::make_shared<apm::PCEstimator>(
                    r, T_c, optional_bootstrap(bootstrap), q);
            }),
            py::arg("r"),
            py::arg("T_c"),
            py::arg("bootstrap") = py::none(),
            py::arg("q") = 0);

    py::class_<apm::PCEstimatorWithFEs,
               apm::PCBase,
               std::shared_ptr<apm::PCEstimatorWithFEs>>(
        module,
        "PCEstimatorWithFEs",
        "Streaming principal-components estimator with outcome fixed effects.")
        .def(
            py::init([](std::size_t r,
                        std::size_t T_c,
                        const py::object& bootstrap,
                        std::size_t q) {
                return std::make_shared<apm::PCEstimatorWithFEs>(
                    r, T_c, optional_bootstrap(bootstrap), q);
            }),
            py::arg("r"),
            py::arg("T_c"),
            py::arg("bootstrap") = py::none(),
            py::arg("q") = 0);

    module.def(
        "align_factors_using_apm",
        [](const py::sequence& cohort_factors,
           const py::sequence& observed_indices,
           const py::object& cohort_weights) {
            const arma::vec weights = cohort_weights.is_none()
                ? arma::vec()
                : to_vec(cohort_weights, "cohort_weights");
            const std::vector<arma::mat> factors =
                to_mats(cohort_factors, "cohort factor matrix");
            const apm::ObservedOutcomeIndices indices =
                to_observed_indices(observed_indices);
            arma::mat result;
            {
                py::gil_scoped_release release;
                result = apm::align_factors_using_apm(factors, indices, weights);
            }
            return to_numpy(result);
        },
        py::arg("cohort_factor_matrices"),
        py::arg("observed_outcome_indices"),
        py::arg("cohort_weights") = py::none(),
        "Align cohort-specific factors into one T x r factor matrix.");

    module.def(
        "compute_aggregated_projection_matrix",
        [](const py::sequence& cohort_factors,
           const py::sequence& observed_indices,
           const py::object& cohort_weights) {
            const arma::vec weights = cohort_weights.is_none()
                ? arma::vec()
                : to_vec(cohort_weights, "cohort_weights");
            const std::vector<arma::mat> factors =
                to_mats(cohort_factors, "cohort factor matrix");
            const apm::ObservedOutcomeIndices indices =
                to_observed_indices(observed_indices);
            arma::mat result;
            {
                py::gil_scoped_release release;
                result = apm::compute_aggregated_projection_matrix(
                    factors, indices, weights);
            }
            return to_numpy(result);
        },
        py::arg("cohort_factor_matrices"),
        py::arg("observed_outcome_indices"),
        py::arg("cohort_weights") = py::none());

    module.def(
        "aggregate_cohort_specific_outcome_fes",
        [](const py::sequence& fixed_effects,
           const py::sequence& observed_indices,
           const py::object& cohort_weights) {
            const std::vector<arma::vec> values =
                to_vecs(fixed_effects, "g_0_c_vec element");
            const apm::ObservedOutcomeIndices indices =
                to_observed_indices(observed_indices);
            const arma::vec weights = cohort_weights.is_none()
                ? arma::vec()
                : to_vec(cohort_weights, "cohort_weights");
            arma::vec result;
            {
                py::gil_scoped_release release;
                result = apm::aggregate_cohort_specific_outcome_fes(
                    values, indices, weights);
            }
            return to_numpy(result);
        },
        py::arg("g_0_c_vec"),
        py::arg("observed_outcome_indices"),
        py::arg("cohort_weights") = py::none());

    module.def(
        "aggregate_cohort_specific_covariate_coefs",
        [](const py::sequence& coefficients, const py::object& cohort_weights) {
            const std::vector<arma::vec> values =
                to_vecs(coefficients, "a_c_vec element");
            const arma::vec weights = cohort_weights.is_none()
                ? arma::vec()
                : to_vec(cohort_weights, "cohort_weights");
            arma::vec result;
            {
                py::gil_scoped_release release;
                result = apm::aggregate_cohort_specific_covariate_coefs(
                    values, weights);
            }
            return to_numpy(result);
        },
        py::arg("a_c_vec"),
        py::arg("cohort_weights") = py::none());

    module.def(
        "impute_outcomes_from_obs_outcomes",
        [](const py::handle& factors,
           const py::handle& observed_indices,
           const py::handle& observed_outcomes,
           const py::object& g_0,
           const py::object& a,
           const py::object& X_c) {
            const arma::mat G = to_mat(factors, "G");
            const arma::uvec T_c = to_uvec(observed_indices, "T_c");
            const arma::vec m_c = to_vec(observed_outcomes, "m_c");
            const std::optional<arma::vec> fixed_effects = optional_vec(g_0, "g_0");
            const std::optional<arma::vec> coefficients = optional_vec(a, "a");
            const std::optional<arma::mat> covariates = optional_mat(X_c, "X_c");
            arma::vec result;
            {
                py::gil_scoped_release release;
                result = impute_from_observed(
                    G, T_c, m_c, fixed_effects, coefficients, covariates);
            }
            return to_numpy(result);
        },
        py::arg("G"),
        py::arg("T_c"),
        py::arg("m_c"),
        py::arg("g_0") = py::none(),
        py::arg("a") = py::none(),
        py::arg("X_c") = py::none(),
        "Complete one cohort from its observed outcomes using zero-based T_c indices.");

    module.def(
        "impute_outcomes_across_cohorts_from_obs_outcomes",
        [](const py::handle& factors,
           const py::sequence& observed_indices,
           const py::sequence& observed_outcomes,
           const py::object& g_0,
           const py::object& a,
           const py::object& X_c_vec) {
            std::optional<std::vector<arma::mat>> covariates;
            if (!X_c_vec.is_none()) {
                covariates = to_mats(
                    py::cast<py::sequence>(X_c_vec), "X_c_vec element");
            }
            const arma::mat G = to_mat(factors, "G");
            const apm::ObservedOutcomeIndices indices =
                to_observed_indices(observed_indices);
            const std::vector<arma::vec> outcomes =
                to_vecs(observed_outcomes, "m_c_vec element");
            const std::optional<arma::vec> fixed_effects = optional_vec(g_0, "g_0");
            const std::optional<arma::vec> coefficients = optional_vec(a, "a");
            arma::mat result;
            {
                py::gil_scoped_release release;
                result = impute_all_from_observed(
                    G,
                    indices,
                    outcomes,
                    fixed_effects,
                    coefficients,
                    covariates);
            }
            return to_numpy(result);
        },
        py::arg("G"),
        py::arg("observed_outcome_indices"),
        py::arg("m_c_vec"),
        py::arg("g_0") = py::none(),
        py::arg("a") = py::none(),
        py::arg("X_c_vec") = py::none(),
        "Complete a C x T cohort outcome matrix from observed entries.");

    module.def(
        "impute_outcomes_across_cohorts",
        [](const py::handle& factors,
           const py::handle& loadings,
           const py::object& g_0,
           const py::object& a,
           const py::object& X_c) {
            std::optional<std::vector<arma::mat>> covariates;
            if (!X_c.is_none()) {
                covariates = to_mats(py::cast<py::sequence>(X_c), "X_c element");
            }
            const arma::mat G = to_mat(factors, "G");
            const arma::mat L = to_mat(loadings, "L");
            const std::optional<arma::vec> fixed_effects = optional_vec(g_0, "g_0");
            const std::optional<arma::vec> coefficients = optional_vec(a, "a");
            arma::mat result;
            {
                py::gil_scoped_release release;
                result = impute_all_from_loadings(
                    G, L, fixed_effects, coefficients, covariates);
            }
            return to_numpy(result);
        },
        py::arg("G"),
        py::arg("L"),
        py::arg("g_0") = py::none(),
        py::arg("a") = py::none(),
        py::arg("X_c") = py::none(),
        "Compute C x T outcomes from factors and cohort loadings.");
}

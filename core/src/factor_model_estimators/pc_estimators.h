#ifndef APM_PC_ESTIMATORS_H
#define APM_PC_ESTIMATORS_H

#include "FactorModelEstimator.h"

namespace apm {

class PCBase : public FactorModelEstimator {
public:
    // Forwarding constructor that calls the base class then initializes
    // outcome_second_moment_mat to a T_c x T_c zero matrix
    explicit PCBase(std::size_t r,
                    std::size_t T_c,
                    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
                    std::size_t q = 0);

protected:
    void add_data_(const arma::uvec& unit_idxs,
                   const arma::mat& Y,
                   const arma::cube& X) override;

    // Helper: combine current second-moment with a new batch using row weights
    static std::pair<arma::mat, double> weighted_combine_second_moment_mats(
        const arma::mat& Y,
        const arma::vec& row_weights,
        const arma::mat& current_second_moment_mat,
        double current_total_weight);

    // Returns n x r matrix whose columns are eigenvectors associated with the
    // largest r eigenvalues of a symmetric PSD matrix S. Assumes S is PSD
    // and symmetric; does not symmetrize.
    static arma::mat top_r_eigenvectors_psd(const arma::mat& S, std::size_t r);

    std::size_t N; // number of observations

    // Accessor for subclasses to read the accumulated second moment matrix
    const arma::mat& outcome_second_moment() const { return outcome_second_moment_mat; }

    // Bootstrap aggregates accessible to subclasses
    arma::vec total_boot_weights;                // length B (if bootstrap present)
    arma::cube boot_outcome_second_moment_mats;  // T_c x T_c x B

private:
    arma::mat outcome_second_moment_mat;         // T_c x T_c
};

class PCEstimator : public PCBase {
public:
    using PCBase::PCBase;

    FactorModelEstimates estimate() override;
};

class PCEstimatorWithFEs : public PCBase {
public:
    explicit PCEstimatorWithFEs(std::size_t r,
                                std::size_t T_c,
                                std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
                                std::size_t q = 0);

    FactorModelEstimates estimate() override;

protected:
    void add_data_(const arma::uvec& unit_idxs,
                   const arma::mat& Y,
                   const arma::cube& X) override;

    static arma::vec weighted_combine_means(
        const arma::mat& Y,
        const arma::vec& row_weights,
        const arma::vec& current_mean,
        double current_total_weight);

private:
    arma::vec outcome_means;
    arma::mat boot_outcome_means; // T_c x B
};

} // namespace apm

#endif // APM_PC_ESTIMATORS_H
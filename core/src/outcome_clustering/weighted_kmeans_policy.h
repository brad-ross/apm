#pragma once

#include <armadillo>

namespace apm {
namespace clustering {

// Weighted Lloyd step mirroring a naive Lloyd implementation, but with weighted centroids.
// Template parameters mirror mlpack's expectations so this can be used as a LloydStepType.
template<typename DistanceType, typename MatType>
class WeightedNaiveKMeans {
public:
    WeightedNaiveKMeans(const MatType& dataset, DistanceType& distance)
        : data(dataset), dist(distance) {}

    // Thread-local pointer to weights (length = number of columns/samples).
    // Must be set by the caller before invoking KMeans::Cluster().
    inline static thread_local const arma::vec* weights_ptr = nullptr;

    // Perform one weighted Lloyd iteration: assign, then weighted centroid update.
    double Iterate(const arma::mat& centroids,
                   arma::mat& newCentroids,
                   arma::Col<size_t>& counts)
    {
        const arma::uword numSamples = data.n_cols;  // columns are samples
        const arma::uword numDims = data.n_rows;
        const arma::uword k = centroids.n_cols;

        newCentroids.zeros(numDims, k);
        counts.zeros(k);

        // Weights must be provided by the caller.
        const arma::vec& w = *weights_ptr;

        // Assignment step (standard Euclidean argmin).
        arma::uvec assignment(numSamples);
        for (arma::uword i = 0; i < numSamples; ++i) {
            arma::vec d = arma::sum(arma::square(centroids.each_col() - data.col(i)), 0).t();
            assignment(i) = d.index_min();
        }

        // Update step: weighted centroids.
        for (arma::uword c = 0; c < k; ++c) {
            arma::uvec idx = arma::find(assignment == c);
            if (idx.n_elem == 0) continue;

            const arma::vec wc = w.elem(idx);
            const double denom = arma::accu(wc);
            if (denom > 0.0) {
                newCentroids.col(c) = (data.cols(idx) * wc) / denom;
            }
            counts(c) = static_cast<size_t>(idx.n_elem); // integer membership for empty-cluster policy
        }

        // Optional: weighted SSE objective value.
        double objective = 0.0;
        for (arma::uword i = 0; i < numSamples; ++i) {
            const arma::uword c = assignment(i);
            objective += w(i) * arma::accu(arma::square(data.col(i) - newCentroids.col(c)));
        }
        return objective;
    }

    size_t DistanceCalculations() const { return 0; }

private:
    const MatType& data;
    DistanceType& dist;
};

} // namespace clustering
} // namespace apm



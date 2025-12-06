# APM: Aggregated Projection Matrix

A high-performance implementation of the Aggregated Projection Matrix (APM) method for counterfactual outcome mean estimation with short, unbalanced panel data proposed in Lei and Ross [(2025+)](https://arxiv.org/abs/2312.07520).

## Overview

APM is a spectral approach for identifying and estimating average counterfactual outcomes under a low-rank factor model with short panel data and general outcome missingness patterns. Applications include event studies and studies of outcomes of "matches" between agents of two types (e.g., workers and firms).

The method identifies all counterfactual outcome means, including those not estimable by existing methods, when a particular graph constructed based on overlaps in observed outcomes between subpopulations is connected. The estimation procedure yields consistent, asymptotically normal estimates under fixed-T (number of outcomes), large-N (sample size) asymptotics.

## Getting Started

The primary interface is through the R package, which you can find in the [`r`](r) directory. See the [R package documentation](r/README.md) for installation and usage instructions.

For developers interested in the core C++ implementation housed in the [`core`](core) directory, see the [C++ core documentation](core/README.md). 
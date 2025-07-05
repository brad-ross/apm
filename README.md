# APM: Aggregated Projection Matrix

A high-performance implementation of the Aggregated Projection Matrix (APM) method for causal inference with panel data.

## Overview

APM is a spectral approach for identifying and estimating average counterfactual outcomes under a low-rank factor model with short panel data and general outcome missingness patterns. Applications include event studies and studies of outcomes of "matches" between agents of two types (e.g., workers and firms).

The method identifies all counterfactual outcome means, including those not estimable by existing methods, when a particular graph constructed based on overlaps in observed outcomes between subpopulations is connected. The estimation procedure yields consistent, asymptotically normal estimates under fixed-T (number of outcomes), large-N (sample size) asymptotics.

## Project Structure

```
apm/
├── README.md                    # This file
├── cpp/                         # C++ core implementation
│   ├── src/
│   ├── tests/
│   └── CMakeLists.txt
├── r/                          # R package
│   ├── src/
│   ├── R/
│   ├── tests/
│   ├── DESCRIPTION
│   └── NAMESPACE
└── .github/                     # CI/CD workflows
    └── workflows/
```

## Getting Started

The primary interface is through the R package. See the [R package documentation](r/README.md) for installation and usage instructions.

For developers interested in the C++ implementation, see the [C++ core documentation](cpp/README.md). 
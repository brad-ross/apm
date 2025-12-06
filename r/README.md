# APM R Package Development Guide

This directory contains the R package for the Aggregated Projection Matrix (APM) method proposed in [Lei and Ross (2025+)](https://arxiv.org/abs/2312.07520), with C++ bindings to the core libr

## Prerequisites

### R Installation
Ensure you have R installed (version 3.5.0 or higher):
```bash
R --version
```

### System Dependencies
- **C++ compiler**: Available through Xcode Command Line Tools (macOS), Rtools (Windows), or build-essential (Linux)

**Note**: Armadillo and Boost are provided through R package dependencies (`RcppArmadillo` and `BH`), so no manual installation of these libraries is required. This makes the package installable across all platforms.

## Package Installation

The package is not yet on CRAN. You can install directly from GitHub:

```r
# Install from GitHub (easiest method)
devtools::install_github("brad-ross/apm", subdir = "r")
```

Or clone the repository locally and install from source:

```bash
git clone https://github.com/your-username/apm.git
```

```r
# Install from local clone
install.packages("path/to/apm/r", repos = NULL, type = "source")

# Or using devtools
devtools::install("path/to/apm/r")
```

## Development Setup

### 1. Install devtools

The `devtools` package is essential for R package development. Install it from within R:

```r
# Option 1: Install from R console
install.packages("devtools")

# Option 2: Install from terminal
R -e 'install.packages("devtools", repos="https://cloud.r-project.org")'
```

### 2. Install Additional Dependencies

```r
# Install required packages for this package
install.packages(c("Rcpp", "RcppArmadillo", "testthat"))
```

## Development Workflow

### Running Tests

The primary way to test your package during development:

```bash
# Navigate to the r directory
cd r

# Run all tests
R -e 'devtools::test()'
```

**Expected output:**
```
ℹ Testing apm
✔ | F W  S  OK | Context
✔ |          2 | apm

══ Results ════════════════════════════════════════════════════════════════════════════════════════════════
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 2 ]
```

### Interactive Development

Load the package for interactive testing:

```r
# Load the package in development mode
devtools::load_all()

# Test functions interactively
get_version()  # Should return "0.1.0"
```

### Other Useful Commands

```r
# Check package for issues
devtools::check()

# Build documentation
devtools::document()

# Clean compiled code (useful if you change C++ code)
devtools::clean_dll()

# Install the package locally
devtools::install()
```

## Package Structure

The codebase is organized with a shared C++ core library and R-specific bindings:

```
apm/
├── core/                      # Shared C++ library
│   ├── src/                   # Core C++ implementation
│   └── tests/                 # C++ unit tests (Catch2)
│
└── r/                         # R package
    ├── DESCRIPTION            # Package metadata and dependencies
    ├── NAMESPACE              # Exported functions
    ├── R/                     # R source files (native R code)
    ├── src/                   # C++ bindings (Rcpp wrappers for core library)
    ├── tests/testthat/        # R unit tests (testthat)
    ├── man/                   # Documentation (.Rd files, auto-generated)
    └── inst/include/          # Headers for package linking
```

### Where to Add New Functionality

| Component | Location | Description |
|-----------|----------|-------------|
| Core C++ logic | `../core/src/` | Platform-independent algorithms and data structures |
| C++ tests | `../core/tests/` | Unit tests for core C++ code |
| R bindings | `src/*_bindings.cpp` | Rcpp wrappers exposing C++ functions to R |
| R functions | `R/` | Pure R code and high-level API functions |
| R tests | `tests/testthat/` | Unit tests for R functions |

**Build configuration updates:**
- When adding new core C++ files, update `../core/CMakeLists.txt` (and `../core/tests/CMakeLists.txt` for test files)
- When adding new R bindings or R source files, update `src/Makevars` (and `src/Makevars.win` for Windows)

## Adding New Functions

### 1. Add to Core Library
First, add your function to the core C++ library:
- Declaration in e.g. `../core/src/apm_core.h`
- Implementation in e.g. `../core/src/apm_core.cpp`

### 2. Add R Binding
Add the R binding in e.g. `src/apm_core_bindings.cpp`:

```cpp
//' Your Function Description
//' 
//' @param input Description of input parameter
//' @return Description of return value
//' @export
// [[Rcpp::export]]
ReturnType your_function_name(InputType input) {
    return apm::your_cpp_function(input);
}
```

### 3. Add Tests
Add tests in e.g. `tests/testthat/test_apm_core.R`:

```r
test_that("your_function_name works", {
  result <- your_function_name(test_input)
  expect_equal(result, expected_output)
})
```

### 4. Test
```bash
R -e 'devtools::test()'
```

## Troubleshooting

### Build Issues
If you encounter compilation errors or change C++ code without changing R code or bindings:

```r
# Clean and rebuild
devtools::clean_dll()
devtools::load_all()
```

### Missing Dependencies
If you get errors about missing packages:

```r
# Install missing dependencies
install.packages(c("Rcpp", "RcppArmadillo", "testthat"))
```

## Development Tips

1. **Always test after changes**: `devtools::test()`
2. **Use interactive development**: `devtools::load_all()`
3. **Check package regularly**: `devtools::check()`
4. **Clean when changing C++**: `devtools::clean_dll()`
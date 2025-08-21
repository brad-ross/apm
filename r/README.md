# APM R Package Development Guide

This directory contains the R package for the Aggregated Projection Matrix (APM) method, with C++ bindings to the core library.

## Prerequisites

### 1. R Installation
Ensure you have R installed (version 3.5.0 or higher):
```bash
R --version
```

### 2. Required System Dependencies
- **C++ compiler**: Already available through Xcode Command Line Tools
- **Armadillo**: Installed via Homebrew
- **Boost**: Installed via Homebrew

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

```
r/
├── DESCRIPTION          # Package metadata
├── NAMESPACE           # Exported functions
├── src/
│   ├── r_bindings.cpp  # C++ bindings to core library
│   └── Makevars        # Build configuration
├── tests/
│   ├── testthat.R      # Test runner
│   └── testthat/
│       └── test-apm-core.R  # Test cases
└── README.md           # This file
```

## Adding New Functions

### 1. Add to Core Library
First, add your function to the core C++ library:
- Declaration in `../core/src/apm_core.h`
- Implementation in `../core/src/apm_core.cpp`

### 2. Add R Binding
Add the R binding in `src/r_bindings.cpp`:

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
Add tests in `tests/testthat/test-apm-core.R`:

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
If you encounter compilation errors:

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

### C++ Compilation Issues
Ensure your core library files exist:
- `../core/src/apm_core.h`
- `../core/src/apm_core.cpp`

The `Makevars` file tells R how to compile these external C++ files.

## Package Installation

For end users (not during development):

```r
# Install from source
install.packages("path/to/apm/r", repos = NULL, type = "source")

# Or using devtools
devtools::install("path/to/apm/r")
```

## Development Tips

1. **Always test after changes**: `devtools::test()`
2. **Use interactive development**: `devtools::load_all()`
3. **Check package regularly**: `devtools::check()`
4. **Clean when changing C++**: `devtools::clean_dll()`
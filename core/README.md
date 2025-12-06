# APM C++ Core Implementation

The C++ core library implements the Aggregated Projection Matrix (APM) method for causal inference with panel data.

## About APM

APM is a spectral approach for identifying and estimating average counterfactual outcomes under a low-rank factor model with short panel data and general outcome missingness patterns. The method identifies all counterfactual outcome means, including those not estimable by existing methods, when a particular graph constructed based on overlaps in observed outcomes between subpopulations is connected.

## Dependencies

### Required
- C++17 compatible compiler
- CMake 3.16+
- Armadillo 9.0+
- Boost 1.70+

### Optional
- Google Test (for testing)

## Dependency Installation

### Ubuntu/Debian
```bash
sudo apt-get install libarmadillo-dev libboost-all-dev cmake build-essential
```

### macOS
```bash
brew install armadillo boost cmake
```

## Building from Source
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Re-running CMake after changing build configuration
If you modify `CMakeLists.txt` (for example, to add new Arrow
components like `arrow-compute`) you only need to regenerate the
build system and rebuild:
```bash
# From the repository root
cd core/build        # or create it first with `mkdir -p core/build`
cmake ..             # re-configure – picks up the new CMake changes
cmake --build . -j$(nproc)   # rebuild everything
```

If you prefer to start with a clean build, simply delete the
`core/build` directory first:
```bash
rm -rf core/build
mkdir core/build && cd core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

## Testing

```bash
cd build
make test
# or
ctest --verbose
```

## Contributing

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
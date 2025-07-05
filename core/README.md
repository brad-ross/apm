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

## Installation

### Ubuntu/Debian
```bash
sudo apt-get install libarmadillo-dev libboost-all-dev cmake build-essential
```

### macOS
```bash
brew install armadillo boost cmake
```

### Building from Source
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Usage

### Basic Example
```cpp
#include "apm.h"
#include <iostream>

int main() {
    // Check library version
    std::cout << "APM version: " << apm::get_version() << std::endl;
    return 0;
}
```

### Compilation
```bash
g++ -std=c++17 -O3 -I/path/to/armadillo/include \
    your_program.cpp -lapm_core -larmadillo -o your_program
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

## License

MIT License - see LICENSE file for details 
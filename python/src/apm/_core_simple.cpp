#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include the APM core headers
#include "../../../core/src/apm_core.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "APM: Aggregated Projection Matrix - Python bindings";
    
    // Just expose the version for now
    m.def("get_version", &apm::get_version, "Get APM library version");
    
    // TODO: Add more functions with proper numpy conversion later
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Include the APM core headers
#include "../../../core/src/apm_core.h"

namespace py = pybind11;

// Simple converter functions (more robust than before)
std::vector<double> numpy_to_vector(py::array_t<double> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    
    std::vector<double> result(buf_info.shape[0]);
    std::memcpy(result.data(), buf_info.ptr, sizeof(double) * buf_info.shape[0]);
    return result;
}

std::vector<std::vector<double>> numpy_to_matrix(py::array_t<double> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    std::vector<std::vector<double>> result(buf_info.shape[0], std::vector<double>(buf_info.shape[1]));
    double* ptr = static_cast<double*>(buf_info.ptr);
    
    for (py::ssize_t i = 0; i < buf_info.shape[0]; ++i) {
        for (py::ssize_t j = 0; j < buf_info.shape[1]; ++j) {
            result[i][j] = ptr[i * buf_info.shape[1] + j];
        }
    }
    return result;
}

py::array_t<double> vector_to_numpy(const std::vector<double>& input) {
    return py::array_t<double>(
        input.size(),
        input.data()
    );
}

arma::mat vector_matrix_to_arma(const std::vector<std::vector<double>>& input) {
    if (input.empty()) return arma::mat();
    arma::mat result(input.size(), input[0].size());
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            result(i, j) = input[i][j];
        }
    }
    return result;
}

arma::vec vector_to_arma(const std::vector<double>& input) {
    arma::vec result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result(i) = input[i];
    }
    return result;
}

std::vector<double> arma_vec_to_vector(const arma::vec& input) {
    std::vector<double> result(input.n_elem);
    for (size_t i = 0; i < input.n_elem; ++i) {
        result[i] = input(i);
    }
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "APM: Aggregated Projection Matrix - Python bindings";
    
    m.def("get_version", &apm::get_version, "Get APM library version");
    
    // Simple impute_outcomes wrapper (factors only version)
    m.def("impute_outcomes", [](
        py::array_t<double> G_py,
        py::list T_c_py,
        py::array_t<double> m_c_py
    ) {
        // Convert inputs
        auto G_vec = numpy_to_matrix(G_py);
        arma::mat G = vector_matrix_to_arma(G_vec);
        
        std::vector<size_t> T_c_vec;
        for (auto item : T_c_py) {
            T_c_vec.push_back(py::cast<size_t>(item));
        }
        arma::uvec T_c(T_c_vec.size());
        for (size_t i = 0; i < T_c_vec.size(); ++i) {
            T_c(i) = T_c_vec[i];
        }
        
        auto m_c_vec = numpy_to_vector(m_c_py);
        arma::vec m_c = vector_to_arma(m_c_vec);
        
        // Call the C++ function
        arma::vec result = apm::impute_outcomes(G, T_c, m_c);
        
        // Convert result back to vector then numpy
        auto result_vec = arma_vec_to_vector(result);
        return vector_to_numpy(result_vec);
    }, "Impute outcomes using factor matrix (simplified version)",
       py::arg("G"), py::arg("T_c"), py::arg("m_c"));
}
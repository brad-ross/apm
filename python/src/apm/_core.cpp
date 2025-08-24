#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Include the APM core headers
#include "../../../core/src/apm_core.h"
#include "../../../core/src/linear_algebra_utils.h"

namespace py = pybind11;

// Helper function to convert between numpy and armadillo
// This is a simplified version - you might want to add more robust conversion
arma::mat numpy_to_arma_mat(py::array_t<double> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    arma::mat result(static_cast<double*>(buf_info.ptr), 
                     buf_info.shape[0], buf_info.shape[1], false, true);
    return result;
}

py::array_t<double> arma_mat_to_numpy(const arma::mat& input) {
    return py::array_t<double>(
        {input.n_rows, input.n_cols},
        {sizeof(double), sizeof(double) * input.n_rows},
        input.memptr()
    );
}

arma::vec numpy_to_arma_vec(py::array_t<double> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    
    arma::vec result(static_cast<double*>(buf_info.ptr), 
                     buf_info.shape[0], false, true);
    return result;
}

py::array_t<double> arma_vec_to_numpy(const arma::vec& input) {
    return py::array_t<double>(
        input.n_elem,
        sizeof(double),
        input.memptr()
    );
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "APM: Aggregated Projection Matrix - Python bindings";
    
    m.def("get_version", &apm::get_version, "Get APM library version");
    
    // Basic align_factors_using_apm wrapper (simplified)
    m.def("align_factors_using_apm", [](
        py::list cohort_factor_matrices_py,
        py::list observed_outcome_indices_py
    ) {
        // Convert Python lists to C++ vectors
        std::vector<arma::mat> cohort_factor_matrices;
        std::vector<arma::uvec> observed_outcome_indices;
        
        for (auto item : cohort_factor_matrices_py) {
            auto np_array = py::cast<py::array_t<double>>(item);
            cohort_factor_matrices.push_back(numpy_to_arma_mat(np_array));
        }
        
        for (auto item : observed_outcome_indices_py) {
            auto np_array = py::cast<py::array_t<unsigned int>>(item);
            py::buffer_info buf_info = np_array.request();
            arma::uvec indices(static_cast<unsigned int*>(buf_info.ptr), 
                              buf_info.shape[0], false, true);
            observed_outcome_indices.push_back(indices);
        }
        
        // Call the C++ function
        arma::mat result = apm::align_factors_using_apm(
            cohort_factor_matrices, observed_outcome_indices
        );
        
        // Convert result back to numpy
        return arma_mat_to_numpy(result);
    }, "Align factor matrices using APM method");
    
    // Basic impute_outcomes wrapper (simplified - factors only version)
    m.def("impute_outcomes", [](
        py::array_t<double> G_py,
        py::array_t<unsigned int> T_c_py,
        py::array_t<double> m_c_py
    ) {
        arma::mat G = numpy_to_arma_mat(G_py);
        
        py::buffer_info T_c_buf = T_c_py.request();
        arma::uvec T_c(static_cast<unsigned int*>(T_c_buf.ptr), 
                       T_c_buf.shape[0], false, true);
        
        arma::vec m_c = numpy_to_arma_vec(m_c_py);
        
        arma::vec result = apm::impute_outcomes(G, T_c, m_c);
        
        return arma_vec_to_numpy(result);
    }, "Impute outcomes using factor matrix");
}
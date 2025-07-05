#include <RcppArmadillo.h>
#include "../../core/src/apm.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

//' Get APM Library Version
//' 
//' Returns the version of the APM library.
//' 
//' @return A character string with the version number
//' @export
// [[Rcpp::export]]
std::string get_version() {
    return apm::get_version();
} 
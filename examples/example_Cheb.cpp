#include <Eigen/Dense>
#include <FMCA/src/util/Macros.h>
#include <FMCA/src/Interpolators/evalChebyshevPolynomials.h>
#include <FMCA/src/Interpolators/evalLegendrePolynomials.h>
#include <FMCA/src/Interpolators/evalMonomials.h>
#include <iostream>

int main() {
  FMCA::Matrix pol;
  FMCA::Matrix x = FMCA::Vector::LinSpaced(7, 0, 1);
  std::cout << x << std::endl;
  std::cout << FMCA::evalMonomials(6, x) << std::endl;
  std::cout << FMCA::evalChebyshevPolynomials(6, x) << std::endl;
  std::cout << FMCA::evalLegendrePolynomials(6, x) << std::endl;
  return 0;
}

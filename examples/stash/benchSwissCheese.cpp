#include <FMCA/Clustering>
#include <iostream>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/print2file.hpp"
#include "../FMCA/src/util/tictoc.hpp"
#include "generateSwissCheese.h"
#include "generateSwissCheeseExp.h"

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};

struct rationalQuadraticKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    const double r = (x - y).norm();
    constexpr double alpha = 0.5;
    constexpr double ell = 1.;
    constexpr double c = 1. / (2. * alpha * ell * ell);
    return std::pow(1 + c * r * r, -alpha);
  }
};

int main(int argc, char *argv[]) {
  const unsigned int dim = atoi(argv[1]);
  const unsigned long long npts = 1e3;
  const unsigned long long npts2 = ((npts + 1) * npts) / 2;
  auto kexp = exponentialKernel();
  auto krq = rationalQuadraticKernel();
  tictoc T;
  T.tic();
  Eigen::MatrixXd P = generateSwissCheese(dim, npts);
  T.toc("generation of Swiss cheese");
  std::cout << "npts2: " << npts2 << std::endl;
  Eigen::MatrixXd retval(npts2, 3);
  T.tic();
  unsigned long long ind = 0;
  for (auto j = 0; j < npts; ++j)
    for (auto i = 0; i <= j; ++i) {
      double r = (P.col(i) - P.col(j)).norm();
      retval.row(ind) << r, kexp(P.col(i), P.col(j)), krq(P.col(i), P.col(j));
      ++ind;
    }
  T.toc("dist comp");
  Bembel::IO::print2m("distStats" + std::to_string(dim) + ".m", "dst", retval,
                      "w");
  return 0;
}

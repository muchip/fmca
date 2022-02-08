// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#define FMCA_CLUSTERSET_
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
/////////////////////////
#include <Eigen/Dense>
#include <FMCA/Samplets>

#include "../FMCA/src/util/Errors.h"
#include "../FMCA/src/util/tictoc.hpp"
#include "generateSwissCheese.h"

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
    constexpr double alpha = 0.2;
    constexpr double ell = 0.01;
    constexpr double c = 0.5 / (alpha * ell * ell);
    return std::pow(1 + c * r * r, -alpha);
  }
};

int main(int argc, char *argv[]) {
  const auto function = exponentialKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = 3;
  const unsigned int dtilde = 2;
  const unsigned int dim = atoi(argv[1]);
  const double threshold = 1e-4;
  tictoc T;
  std::fstream file;
  file.open("output" + std::to_string(dim) + ".txt", std::ios::out);
  file << "i          m           n       nz(A)";
  file << "         mem         err\n";
  for (auto i = 2; i < 8; ++i) {
    file << i << "\t";
    const unsigned npts = std::pow(10, i);
    std::cout << "N: " << npts << std::endl;
    // const Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
    const Eigen::MatrixXd P = generateSwissCheese(dim, npts, 100);

    const FMCA::NystromMatrixEvaluator<FMCA::H2SampletTree, exponentialKernel>
        nm_eval(P, function);
    T.tic();
    FMCA::H2SampletTree ST(P, 1, dtilde, mp_deg);
    T.toc("tree setup: ");
    FMCA::symmetric_compressor_impl<FMCA::H2SampletTree> symComp;
    T.tic();
    symComp.compress(ST, nm_eval, eta, threshold);
    T.toc("symmetric compressor: ");

    {
      Eigen::SparseMatrix<double> Ssym(npts, npts);
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      double nrm = 0;
      const auto &trips = symComp.pattern_triplets();
      file << std::setw(10) << std::setprecision(6) << npts << "\t";
      file << std::setw(10) << std::setprecision(6) << npts << "\t";
      file << std::setw(10) << std::setprecision(6)
           << 2 * std::ceil(double(trips.size()) / npts) - 1 << "\t";
      Ssym.setFromTriplets(trips.begin(), trips.end());
      file << std::setw(10) << std::setprecision(6)
           << 3 * (2 * double(trips.size()) - npts) * sizeof(double) / 1e9
           << "\t";
      std::cout << "nz(S): " << 2 * std::ceil(double(trips.size()) / npts) - 1
                << std::endl;
      std::cout << "memory: "
                << 3 * (2 * double(trips.size()) - npts) * sizeof(double) / 1e9
                << "GB\n";
      for (auto i = 0; i < 100; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, ST.indices(), function, index);
        x = ST.sampletTransform(x);
        y2 = Ssym * x +
             Ssym.triangularView<Eigen::StrictlyUpper>().transpose() * x;
        y2 = ST.inverseSampletTransform(y2);
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl;
      file << std::setw(10) << std::setprecision(6) << err << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

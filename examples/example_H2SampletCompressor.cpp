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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/src/util/Errors.h"
#include "../FMCA/src/util/print2file.hpp"
#include "../FMCA/src/util/tictoc.hpp"

#include <FMCA/Samplets>

#define DIM 2
#define MPOLE_DEG 3
#define DTILDE 3
#define LEAFSIZE 100

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-0.1 * (x - y).norm());
  }
};

int main() {
  const auto function = exponentialKernel();
  const double threshold = 1e-6;
  const double eta = 0.8;

  for (auto i = 15; i <= 15; ++i) {
    auto npts = 1 << i;
    std::cout << npts << std::endl;
    const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, npts);
    const FMCA::NystromMatrixEvaluator<FMCA::H2SampletTree, exponentialKernel>
        nm_eval(P, function);
    tictoc T;
    T.tic();
    FMCA::H2SampletTree ST(P, LEAFSIZE, DTILDE, MPOLE_DEG);
    T.toc("tree setup: ");
    FMCA::symmetric_compressor_impl<FMCA::H2SampletTree> symComp;
    FMCA::unsymmetric_compressor_impl<FMCA::H2SampletTree> Comp;
    T.tic();
    Comp.compress(ST, nm_eval, eta, threshold);
    T.toc("unsymmetric compressor: ");
    T.tic();
    symComp.compress(ST, nm_eval, eta, threshold);
    T.toc("symmetric compressor: ");

    {
      Eigen::SparseMatrix<double> S(npts, npts);
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      const auto &trips = Comp.pattern_triplets();
      std::cout << "nnz(S)=" << trips.size() << std::endl;
      S.setFromTriplets(trips.begin(), trips.end());
      for (auto i = 0; i < 10; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, ST.indices(), function, index);
        x = ST.sampletTransform(x);
        y2 = S * x;
        y2 = ST.inverseSampletTransform(y2);
        err += (y1 - y2).norm() / y1.norm();
      }
      err /= 10;
      std::cout << "compression error unsymmetric: " << err << std::endl;
    }
    {
      Eigen::SparseMatrix<double> Ssym(npts, npts);
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      const auto &trips = symComp.pattern_triplets();
      std::cout << "nnz(S)=" << trips.size() << std::endl;
      Ssym.setFromTriplets(trips.begin(), trips.end());
      for (auto i = 0; i < 10; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, ST.indices(), function, index);
        x = ST.sampletTransform(x);
        y2 = Ssym * x +
             Ssym.triangularView<Eigen::StrictlyUpper>().transpose() * x;
        y2 = ST.inverseSampletTransform(y2);
        err += (y1 - y2).norm() / y1.norm();
      }
      err /= 10;
      std::cout << "compression error symmetric: " << err << std::endl;
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

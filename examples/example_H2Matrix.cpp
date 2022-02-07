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
#include <iostream>

#include <Eigen/Dense>
#include <FMCA/H2Matrix>
#include <FMCA/src/util/tictoc.hpp>

#include "../FMCA/src/H2Matrix/backward_transform_impl.h"
#include "../FMCA/src/H2Matrix/forward_transform_impl.h"
#include "../FMCA/src/H2Matrix/matrix_vector_product_impl.h"
#include "../FMCA/src/util/Errors.h"

#define DIM 2
#define NPTS 500000

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-5 * (x - y).norm() / sqrt(DIM)) * x(1);
  }
};

int main() {
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  const FMCA::H2ClusterTree H2CT(P, 20, 5);
  const double eta = 0.4;
  tictoc T;

  T.tic();
  FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(P, H2CT, function, eta);
  T.toc("matrix setup: ");
  {
    Eigen::VectorXd x(NPTS), y1(NPTS), y2(NPTS);
    double err = 0;
    for (auto i = 0; i < 10; ++i) {
      unsigned int index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      y1 = FMCA::matrixColumnGetter(P, H2CT.indices(), function, index);
      y2 = matrix_vector_product_impl(H2mat, x);
      err += (y1 - y2).norm() / y1.norm();
    }
    err /= 10;
    std::cout << "compression error: " << err << std::endl;
    H2mat.get_statistics();
  }
  std::cout << std::string(60, '-') << std::endl;

  return 0;
}

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

#include "../FMCA/src/H2Matrix/forward_transform_impl.h"
#include "../FMCA/src/H2Matrix/backward_transform_impl.h"

#define DIM 5

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-5 * (x - y).norm() / sqrt(DIM));
  }
};

int main() {
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, 20000);
  const FMCA::H2ClusterTree H2CT(P, 60, 3);
  tictoc T;
  for (double eta = 0.8; eta >= 0.8; eta -= 0.2) {
    std::cout << "eta= " << eta << std::endl;
    T.tic();
    FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(P, H2CT, function, eta);
    T.toc("matrix setup: ");
    T.tic();
    auto tvec = forward_transform_impl(H2mat, P.transpose());
    T.toc("fw trafo:");
    T.tic();
    auto ttvec = backward_transform_impl(H2mat, tvec);
    T.toc("bw trafo:");
    H2mat.get_statistics();
    Eigen::MatrixXd K(P.cols(), P.cols());
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) = function(P.col(H2CT.indices()[i]), P.col(H2CT.indices()[j]));
    std::cout << "H2-matrix compression error: "
              << (K - H2mat.full()).norm() / K.norm() << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }

  return 0;
}

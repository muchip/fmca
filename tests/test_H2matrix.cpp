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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "TestParameters.h"

int main() {
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  const FMCA::H2ClusterTree H2CT(P, LEAFSIZE, MPOLE_DEG);
  const FMCA::NystromMatrixEvaluator<FMCA::H2ClusterTree, exponentialKernel>
      nm_eval(P, function);
  for (double eta = 0.8; eta >= 0; eta -= 0.2) {
    std::cout << "eta= " << eta << std::endl;
    FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(H2CT, nm_eval, eta);
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

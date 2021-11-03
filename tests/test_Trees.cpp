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

#include "../FMCA/Samplets"
#include "../FMCA/src/util/tictoc.hpp"
#include "TestParameters.h"

int main() {
  tictoc T;
  for (auto J = 4; J <= 24; J += 2) {
    const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, 1 << J);
    std::cout << "npts: " << (1 << J) << std::endl;
    T.tic();
    const FMCA::ClusterT CT(P, LEAFSIZE);
    T.toc("cluster tree: ");
    T.tic();
    FMCA::H2ClusterTree H2CT(P, LEAFSIZE, MPOLE_DEG);
    T.toc("H2-cluster tree: ");
    T.tic();
    const FMCA::SampletTreeQR ST(P, LEAFSIZE, DTILDE);
    T.toc("samplet tree: ");
    T.tic();
    FMCA::H2SampletTree H2ST(P, LEAFSIZE, DTILDE, MPOLE_DEG);
    T.toc("H2 samplet tree: ");
    std::cout << std::string(60, '-') << std::endl;
  }

  return 0;
}

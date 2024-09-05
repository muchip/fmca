// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/src/LowRankApproximation/FALKON.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 1000000
#define DIM 3

int main() {
  FMCA::Tictoc T;
  FMCA::FALKON falkon;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "FALKON" << std::endl;
  // check for a small setting that the preconditioner in the M=n
  // case gives the identity
  {
    const FMCA::Matrix Psmall = FMCA::Matrix::Random(DIM, 1000);
    const FMCA::CovarianceKernel kernel_small("EXPONENTIAL", 2.);
    falkon.init(kernel_small, Psmall, 1000, 1e-6);
    FMCA::Matrix I = FMCA::Matrix::Identity(1000, 1000);
    std::cout << "precond. accuracy (M=n case): "
              << (falkon.BTBTimesVector(I) - I).norm() / I.norm() << std::endl;
  }
  // now solve a large system
  {
    const FMCA::CovarianceKernel kernel("GAUSSIAN", 2.);
    const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
    const FMCA::Vector data = FMCA::Vector::Ones(NPTS);
    T.tic();
    falkon.init(kernel, P, 1000, 1e-10);
    T.toc("elapsed time:                ");
    T.tic();
    FMCA::Vector alpha = falkon.computeAlpha(data, 10);
    T.toc("elapsed time:                ");
    std::cout << "relative error:               "
              << (falkon.matrixKPC() * alpha - data).norm() / data.norm()
              << std::endl;
  }
  return 0;
}

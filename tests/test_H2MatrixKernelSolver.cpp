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
#include "../FMCA/H2Matrix"
#include "../FMCA/src/KernelInterpolation/H2MatrixKernelSolver.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 1000
#define DIM 3
#define MPOLE_DEG 3

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 2.);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  FMCA::H2MatrixKernelSolver h2ks(function, P, 3, 0.5);
  h2ks.compress(P);
  h2ks.K().statistics();
  std::cout << "compression error: " << h2ks.compressionError(P) << std::endl;
  FMCA::Vector rhs(NPTS);
  rhs.setRandom();
  h2ks.solveIteratively(rhs);
  return 0;
}

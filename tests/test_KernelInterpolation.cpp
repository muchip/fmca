// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
// #define EIGEN_DONT_PARALLELIZE
////////////////////////////////////////////////////////////////////////////////
#define CHOLMOD_SUPPORT
#include <iostream>

#include "../FMCA/KernelInterpolation"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 10000
#define DIM 2

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1.);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Matrix Peval =
      0.5 * (FMCA::Matrix::Random(DIM, 2000).array() + 1);
  const FMCA::Index dtilde = 6;

  const FMCA::Scalar threshold = 1e-8;
  const FMCA::Scalar eta = 0.5;
  T.tic();
  FMCA::SampletKernelSolver SKS(function, P, dtilde, eta, threshold);
  FMCA::MultipoleFunctionEvaluator MFE(function, P, Peval, eta, 10);
  T.toc("compression time:            ");
  std::cout << "Compression error:            " << SKS.compressionError(P)
            << std::endl;
  T.tic();
  SKS.factorize();
  T.toc("factorization time:          ");
  T.tic();
  FMCA::Vector data(P.cols());
  data.setOnes();
  FMCA::Matrix sol = SKS.solve(data);
  FMCA::Matrix eval = MFE.evaluate(P, Peval, sol);
  FMCA::Matrix ref(Peval.cols(), 1);
  ref.setOnes();
  std::cout << "oos error:                    "
            << (eval - ref).norm() / ref.norm() << std::endl;
  T.toc("solution/evaluation time:    ");
}

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
#include "../FMCA/src/LowRankApproximation/NystromApproximation.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 1000000
#define DIM 2

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel kernel("MATERN52", 1.);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  T.tic();
  FMCA::NystromApproximation NystApp;
  NystApp.compute(kernel, P, 34);
  T.toc("elapsed time:                ");
  {
    FMCA::Vector colOp;
    FMCA::Vector colL;
    FMCA::Scalar error = 0;
    FMCA::Scalar fnorm2 = 0;
    Eigen::Index sampleCol = 0;
    std::srand(0);
    // compare random columns of C to the respective ones of L * L'
    for (auto i = 0; i < 100; ++i) {
      sampleCol = std::rand() % P.cols();
      colOp = kernel.eval(P, P.col(sampleCol));
      colL = NystApp.matrixL() * NystApp.matrixL().row(sampleCol).transpose();
      error += (colOp - colL).squaredNorm();
      fnorm2 += colOp.squaredNorm();
    }
    std::cout << "sampled Frobenius error:      " << sqrt(error / fnorm2)
              << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;
  return 0;
}

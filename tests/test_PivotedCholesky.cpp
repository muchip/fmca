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
#include "../FMCA/src/LowRankApproximation/PivotedCholesky.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 1000000
#define DIM 3

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel kernel("GAUSSIAN", 2.);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  T.tic();
  FMCA::PivotedCholesky pivChol;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "Pivoted Cholesky decomposition" << std::endl;
  pivChol.compute(kernel, P, 1. / NPTS);
  pivChol.computeBiorthogonalBasis();
  T.toc("elapsed time:                ");
  std::cout << "rank:                         " << pivChol.matrixL().cols()
            << std::endl;
  {
    FMCA::Vector colOp;
    FMCA::Vector colL;
    FMCA::Scalar error = 0;
    FMCA::Scalar fnorm2 = 0;
    Eigen::Index sampleCol = 0;
    std::srand(std::time(NULL));
    // compare random columns of C to the respective ones of L * L'
    for (auto i = 0; i < 100; ++i) {
      sampleCol = std::rand() % P.cols();
      colOp = kernel.eval(P, P.col(sampleCol));
      colL = pivChol.matrixL() * pivChol.matrixL().row(sampleCol).transpose();
      error += (colOp - colL).squaredNorm();
      fnorm2 += colOp.squaredNorm();
    }
    std::cout << "sampled Frobenius error:      " << sqrt(error / fnorm2)
              << std::endl;
  }
  {
    const FMCA::Matrix BTL = pivChol.matrixB().transpose() * pivChol.matrixL();
    std::cout << "dual basis error:             "
              << (BTL - FMCA::Matrix::Identity(BTL.rows(), BTL.cols())).norm() /
                     std::sqrt(BTL.rows())
              << std::endl;
  }
  {
    FMCA::Matrix Ctrs(P.rows(), pivChol.indices().size());
    for (FMCA::Index i = 0; i < pivChol.indices().size(); ++i)
      Ctrs.col(i) = P.col(pivChol.indices()(i));
    std::cout << "Newton basis weights error:   "
              << (kernel.eval(P, Ctrs) * pivChol.matrixU() - pivChol.matrixL())
                         .norm() /
                     pivChol.matrixL().norm()
              << std::endl;
    FMCA::Matrix specBasis =
        kernel.eval(P, Ctrs) * pivChol.spectralBasisWeights();
    FMCA::Matrix Lambda = pivChol.eigenvalues().asDiagonal();
    std::cout << "Spectral basis weights error: "
              << (specBasis.transpose() * specBasis - Lambda).norm() /
                     Lambda.norm()
              << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "Orthogonal matching pursuit" << std::endl;
  FMCA::Vector f = kernel.eval(P, P.col(0)) + kernel.eval(P, P.col(1)) +
                   kernel.eval(P, P.col(2)) + kernel.eval(P, P.col(3));
  T.tic();
  pivChol.computeOMP(kernel, P, f, 1. / NPTS);
  T.toc("elapsed time:                ");
  std::cout << "rank:                         " << pivChol.matrixL().cols()
            << std::endl;
  {
    const FMCA::Matrix BTL = pivChol.matrixB().transpose() * pivChol.matrixL();
    std::cout << "dual basis error:             "
              << (BTL - FMCA::Matrix::Identity(BTL.rows(), BTL.cols())).norm() /
                     std::sqrt(BTL.rows())
              << std::endl;
  }
  {
    FMCA::Matrix Ctrs(P.rows(), pivChol.indices().size());
    for (FMCA::Index i = 0; i < pivChol.indices().size(); ++i)
      Ctrs.col(i) = P.col(pivChol.indices()(i));
    std::cout << "Newton basis weights error:   "
              << (kernel.eval(P, Ctrs) * pivChol.matrixU() - pivChol.matrixL())
                         .norm() /
                     pivChol.matrixL().norm()
              << std::endl;
    FMCA::Matrix specBasis =
        kernel.eval(P, Ctrs) * pivChol.spectralBasisWeights();
    FMCA::Matrix Lambda = pivChol.eigenvalues().asDiagonal();
    std::cout << "Spectral basis weights error: "
              << (specBasis.transpose() * specBasis - Lambda).norm() /
                     Lambda.norm()
              << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;
  return 0;
}

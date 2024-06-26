// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <fstream>
#include <iostream>
//
#include <Eigen/Dense>

#include "../FMCA/Clustering"
#include "../FMCA/CovarianceKernel"
#include "../FMCA/HMatrix"
#include "../FMCA/src/util/Tictoc.h"

////////////////////////////////////////////////////////////////////////////////
using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using HMatrix = FMCA::HMatrix<H2ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
///
int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 1000;
  const FMCA::Index dim = 2;
  const FMCA::Index K = 2;
  const FMCA::Scalar ridge_parameter = 0 * npts;
  const FMCA::CovarianceKernel kernel("MaternNu", .1, 1., 0.5);
  FMCA::Matrix P = FMCA::Matrix::Random(dim, npts);
  const Moments mom(P, 0);
  const MatrixEvaluator mat_eval(mom, kernel);
  std::cout << std::string(72, '-') << std::endl;
  T.tic();
  const H2ClusterTree CT(mom, 20, P);
  std::cout << "Cluster splitter:             "
            << FMCA::internal::traits<H2ClusterTree>::Splitter::splitterName()
            << std::endl;
  T.toc("cluster tree:                ");
  T.tic();
  Eigen::SparseMatrix<FMCA::Scalar> invsqrtK =
      FMCA::sparseKernelMatrixInverseSqrt(kernel, CT, P, K, ridge_parameter);
  T.toc("sparse inverse sqrt:         ");
  std::cout << "anz:                          " << invsqrtK.nonZeros() / npts
            << std::endl;
  T.tic();
  std::cout << std::string(72, '-') << std::endl;
  const HMatrix hmat(CT, mat_eval, 0.8, 1e-6);
  T.toc("elapsed time H-matrix:       ");
  hmat.statistics();
  {
    FMCA::Matrix X(npts, 100), Y1(npts, 100), Y2(npts, 100);
    X.setZero();
    for (auto i = 0; i < 100; ++i) {
      FMCA::Index index = rand() % P.cols();
      FMCA::Vector col = kernel.eval(P, P.col(CT.indices()[index]));
      Y1.col(i) =
          col(Eigen::Map<const FMCA::iVector>(CT.indices(), CT.block_size()));
      X(index, i) = 1;
    }
    Y2 = hmat * X;
    FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
    std::cout << "compression error Hmatrix:    " << err << std::endl;
    std::cout << std::string(72, '-') << std::endl;
  }

  FMCA::Matrix X(npts, 100), Y1(npts, 100), Y2(npts, 100);
  X.setRandom();
  FMCA::Matrix Y0 = (invsqrtK.transpose() * X);
  Y1 = hmat * Y0 + ridge_parameter * Y0;
  Y2 = invsqrtK * Y1.eval();
  std::cout << "sym inverse error:            " << (Y2 - X).norm() / X.norm()
            << std::endl;
  std::cout << std::string(72, '-') << std::endl;

  return 0;
}

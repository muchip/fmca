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
#include "../FMCA/HMatrix"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using HMatrix = FMCA::HMatrix<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("GAUSSIAN", 2.);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  const FMCA::ClusterTree ct(P, 10);
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  const Moments mom(P, 0);
  const MatrixEvaluator mat_eval(mom, function);
  FMCA::Scalar eta = 10;
  {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    HMatrix hmat;
    hmat.computeHMatrix(ct, ct, mat_eval, eta, 1e-5);
    T.toc("elapsed time:                ");
    hmat.statistics();
    {
      FMCA::Matrix X(NPTS, 10), Y1(NPTS, 10), Y2(NPTS, 10);
      X.setZero();
      X.setZero();
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % P.cols();
        FMCA::Vector col = function.eval(P, P.col(ct.indices()[index]));
        Y1.col(i) =
            col(Eigen::Map<const FMCA::iVector>(ct.indices(), ct.block_size()));
        X(index, i) = 1;
      }
      std::cout << "set test data" << std::endl;
      T.tic();
      Y2 = hmat * X;
      FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
      std::cout << "compression error:            " << err << std::endl;
    }
    T.toc("elapsed time:                ");
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

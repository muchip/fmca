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
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2
#define MPOLE_DEG 3

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 2.);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  const Moments mom(P, MPOLE_DEG);
  T.tic();
  H2ClusterTree ct(mom, 0, P);
  T.toc("H2 cluster tree:");
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ct, mom);
  const MatrixEvaluator mat_eval(mom, function);
  for (FMCA::Scalar eta = 0.8; eta >= 0.1; eta *= 0.5) {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    H2Matrix hmat;
    hmat.computePattern(ct, ct, eta);
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
      Y2 = hmat.action(mat_eval, X);
      FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
      std::cout << "compression error:            " << err << std::endl;
    }
    T.toc("elapsed time:                ");
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

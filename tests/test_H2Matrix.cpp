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

#define NPTS 10000
#define DIM 3
#define MPOLE_DEG 3

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using MatrixEvaluatorUS =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree, FMCA::CompareCluster>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 2.);
  const FMCA::Matrix Pr = FMCA::Matrix::Random(DIM, 2 * NPTS);
  const FMCA::Matrix Pc = FMCA::Matrix::Random(DIM, NPTS);

  const Moments momr(Pr, MPOLE_DEG);
  const Moments momc(Pc, MPOLE_DEG);

  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.tic();
  H2ClusterTree ctr(momr, 0, Pr);
  H2ClusterTree ctc(momc, 0, Pc);
  T.toc("H2 cluster tree:");
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ctr,
                                                                      momr);
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ctc,
                                                                      momc);
  const MatrixEvaluatorUS mat_eval(momr, momc, function);
  for (FMCA::Scalar eta = 0.8; eta >= 0.1; eta *= 0.5) {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    H2Matrix hmat;
    hmat.computePattern(ctr, ctc, eta);
    T.toc("elapsed time:                ");
    hmat.statistics();

    {
      FMCA::Matrix X(NPTS, 10), Y1(2 * NPTS, 10), Y2(2 * NPTS, 10);
      X.setZero();
      X.setZero();
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % Pc.cols();
        FMCA::Vector col = function.eval(Pr, Pc.col(ctc.indices()[index]));
        Y1.col(i) = col(
            Eigen::Map<const FMCA::iVector>(ctr.indices(), ctr.block_size()));
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

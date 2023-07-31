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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 100
#define MPOLE_DEG 4

using Interpolator = FMCA::WeightedTotalDegreeInterpolator;
using Moments = FMCA::WeightedNystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1);
  FMCA::Vector b(DIM);
  std::vector<FMCA::Scalar> w(DIM);
  for (FMCA::Index i = 0; i < DIM; ++i) {
    b(i) = std::pow(FMCA::Scalar(i + 1), -4.);
    w[i] = log(2. / b(i) + sqrt(1 + 4. / b(i) / b(i)));
  }
  for (FMCA::Index i = 0; i < DIM; ++i) w[i] /= w[0];
  std::cout << "dimension: " << DIM << std::endl;
  for (FMCA::Index npts : {1e4, 1e5, 1e6, 1e7}) {
    std::cout << "npts: " << npts << std::endl;
    FMCA::Matrix P(DIM, npts);
    P.setRandom();
    P = b.asDiagonal() * P;
    const Moments mom(P, w, MPOLE_DEG);

    H2ClusterTree ct(mom, 0, P);
    std::vector<FMCA::Matrix> bbs;
    for (const auto &node : ct) {
      if (!node.nSons()) bbs.push_back(node.bb());
    }
    // FMCA::IO::plotBoxes("boxes.vtk", bbs);
    FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ct,
                                                                        mom);
    const MatrixEvaluator mat_eval(mom, function);
    for (FMCA::Scalar eta : {0.8, 0.5}) {
      std::cout << "eta:                          " << eta << std::endl;
      T.tic();
      const H2Matrix hmat(ct, mat_eval, eta);
      T.toc("elapsed time:                ");
      hmat.statistics();
      {
        FMCA::Matrix X(npts, 10), Y1(npts, 10), Y2(npts, 10);
        X.setZero();
        for (auto i = 0; i < 10; ++i) {
          FMCA::Index index = rand() % P.cols();
          FMCA::Vector col = function.eval(P, P.col(ct.indices()[index]));
          Y1.col(i) = col(ct.indices());
          X(index, i) = 1;
        }
        Y2 = hmat * X;
        FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
        std::cout << "compression error:            " << err << std::endl;
        std::cout << std::string(60, '-') << std::endl;
      }
    }
  }
  return 0;
}

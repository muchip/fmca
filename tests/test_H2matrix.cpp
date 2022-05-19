// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/H2Matrix"
#include "TestParameters.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator =
    FMCA::NystromMatrixEvaluator<Moments, exponentialKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  const auto function = exponentialKernel();
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  const Moments mom(P, MPOLE_DEG);
  H2ClusterTree ct(mom, 0, P);
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ct, mom);
  const MatrixEvaluator mat_eval(mom, function);
  for (FMCA::Scalar eta = 0.8; eta >= 0; eta -= 0.2) {
    std::cout << "eta= " << eta << std::endl;
    const H2Matrix hmat(ct, mat_eval, eta);
    hmat.get_statistics();
    FMCA::Matrix K(P.cols(), P.cols());
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) = function(P.col(ct.indices()[i]), P.col(ct.indices()[j]));
    std::cout << "H2-matrix compression error: "
              << (K - hmat.full()).norm() / K.norm() << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }

  return 0;
}

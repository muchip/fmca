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
#define FMCA_CLUSTERSET_
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Samplets"
#include "TestParameters.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator =
    FMCA::NystromMatrixEvaluator<Moments, exponentialKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  const auto function = exponentialKernel();
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  const FMCA::Scalar threshold = 1e-15;
  const MatrixEvaluator nm_eval(P, function);

  for (FMCA::Index dtilde = 1; dtilde <= 4; ++dtilde) {
    for (FMCA::Scalar eta = 1; eta >= 0; eta -= 0.5) {
      std::cout << "dtilde= " << dtilde << " eta= " << eta << std::endl;
      const Moments mom(P, 6);
      const MatrixEvaluator mat_eval(mom, function);
      const SampletMoments samp_mom(P, dtilde - 1);
      H2SampletTree hst(mom, samp_mom, 0, P);
      FMCA::symmetric_compressor_impl<H2SampletTree> Scomp;
      Scomp.compress(hst, mat_eval, eta, threshold);
      const auto &trips = Scomp.pattern_triplets();
      FMCA::Matrix K;
      mat_eval.compute_dense_block(hst, hst, &K);
      hst.sampletTransformMatrix(K);
      Eigen::SparseMatrix<double> S(K.rows(), K.cols());
      S.setFromTriplets(trips.begin(), trips.end());
      Eigen::MatrixXd K2 = S;
      K2 += Eigen::MatrixXd(
          K2.triangularView<Eigen::StrictlyUpper>().transpose());
      std::cout << "compression error: " << (K2 - K).norm() / K.norm()
                << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }
  }
  return 0;
}

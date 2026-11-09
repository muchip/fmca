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
// #define EIGEN_DONT_PARALLELIZE
#include <FMCA/src/util/Tictoc.h>

#include <Eigen/Dense>
#include <FMCA/Kernel>
#include <FMCA/Samplets>
#include <iostream>

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1);
  const FMCA::Index m = 7896;
  const FMCA::Index n = 9896;
  const FMCA::Index dim = 2;
  const FMCA::Matrix P1 = 0.5 * (FMCA::Matrix::Random(dim, m).array() + 1);
  const FMCA::Matrix P2 = 0.5 * (FMCA::Matrix::Random(dim, n).array() + 1);
  const FMCA::Scalar threshold = 1e-10;
  const FMCA::Index dtilde = 6;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom1(P1, mpole_deg);
  const Moments mom2(P2, mpole_deg);
  const usMatrixEvaluator mat_eval(mom1, mom2, function);
  const SampletMoments samp_mom1(P1, dtilde - 1);
  const SampletMoments samp_mom2(P2, dtilde - 1);
  H2SampletTree hst1(mom1, samp_mom1, 0, P1);
  H2SampletTree hst2(mom2, samp_mom2, 0, P2);
  std::cout
      << "This tests the old unsymmetric compressor against the new dag based. "
         "As a different scheduling is used, the results vary and difference "
         "gets smaller for eta->0. Higher similarity is achieved for "
         "increasing dtilde. The use of SampletMatrixCompressorUnsymmetric is "
         "discouraged and it will vanish in a future version"
      << std::endl;
  for (double eta = 1.2; eta >= 0.0; eta -= 0.2) {
    std::cout << "dtilde:                       " << dtilde << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    FMCA::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst1, hst2, eta, threshold);
    T.toc("unsymmetric planner:         ");
    T.tic();
    Scomp.compress(mat_eval);
    T.toc("unsymmetric compressor:      ");
    T.tic();
    const auto &trips = Scomp.triplets();
    T.toc("triplets:                    ");
    std::cout << "anz:                          "
              << std::round(trips.size() / FMCA::Scalar(m)) << std::endl;
    T.tic();
    FMCA::SparseMatrix S1(m, n);
    S1.setFromTriplets(trips.begin(), trips.end());
    FMCA::SampletMatrixCompressor<H2SampletTree> sScomp;
    sScomp.init(hst1, hst2, eta, threshold);
    T.toc("dag planner:                 ");
    T.tic();
    sScomp.compress(mat_eval);
    T.toc("dag compressor:              ");
    T.tic();
    const auto &strips = sScomp.triplets();
    T.toc("triplets:                    ");
    std::cout << "anz:                          "
              << std::round(strips.size() / FMCA::Scalar(m)) << std::endl;
    FMCA::SparseMatrix S2(m, n);
    S2.setFromTriplets(strips.begin(), strips.end());
    std::cout << "compression error:            "
              << (S1 - S2).norm() / S1.norm() << std::endl
              << std::flush;
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

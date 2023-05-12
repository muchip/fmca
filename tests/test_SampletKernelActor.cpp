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
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/SampletKernelActor.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2
#define MPOLE_DEG 6

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("matern32", 1);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar threshold = 1e-10;
  const FMCA::Index dtilde = 4;
  const FMCA::Scalar eta = 1. / DIM;
  const Moments mom(P, MPOLE_DEG);
  const usMatrixEvaluator mat_eval(mom, mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  H2ClusterTree hct(mom, hst);
  auto it = hst.begin();
  auto it2 = hct.begin();
  while (it != hst.end() && it2 != hct.end()) {
    assert(it->block_id() == it2->block_id() && "block id mismatch");
    assert(it->indices() == it2->indices() && "index mismatch");
    ++it;
    ++it2;
  }
  assert(it == hst.end() && it2 == hct.end() && "tree size mismatch");
  T.tic();
  FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
  Scomp.init(hst, hst, eta, threshold);
  T.toc("unsymmetric planner:         ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("unsymmetric compressor:      ");
  T.tic();
  const auto &trips = Scomp.triplets();
  T.toc("triplets:                    ");
  std::cout << "anz:                          "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  FMCA::SparseMatrix S(NPTS, NPTS);
  S.setFromTriplets(trips.begin(), trips.end());
  FMCA::internal::compute_cluster_bases_impl::compute(hst, mom);

  T.tic();
  const FMCA::SampletKernelActor<usMatrixEvaluator, H2ClusterTree,
                                 H2SampletTree>
      hact(mat_eval, hst, hst, MPOLE_DEG, eta);
  T.toc("set up actor:");
  FMCA::Matrix X(NPTS, 100);
  X.setRandom();
  T.tic();
  //FMCA::Matrix TX = hst.inverseSampletTransform(X);
  //FMCA::Matrix sX = TX;
  FMCA::Matrix Y = hact.action(X);
  //FMCA::Matrix sY = Y;
  //sY = hst.sampletTransform(sY);
  T.toc("compute action:");
  std::cout << "error: " << (S * X - Y).norm() / X.norm() << std::endl;
  return 0;
}

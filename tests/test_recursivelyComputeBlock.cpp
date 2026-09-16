// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
// #define EIGEN_DONT_PARALLELIZE
#define FMCA_VERBOSE
#include "../FMCA/src/util/Macros.h"
//
#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/recursivelyComputeBlock.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 10000
#define DIM 2

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1.);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar eta = 0.05;
  const FMCA::Index dtilde = 3;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  const FMCA::Matrix K = function.eval(P, P);
  FMCA::Matrix KSig = hst.toClusterOrder(K);
  KSig = hst.toClusterOrder(KSig.transpose());
  hst.sampletTransformMatrix(KSig);

  // choose some cluster pair and test the recurisvelyComputeBlock function
  {
    H2SampletTree &c1 = hst.sons(0).sons(1).sons(0).sons(1);
    H2SampletTree &c2 = hst.sons(0).sons(0);
    FMCA::Matrix K00 =
        FMCA::internal::recursivelyComputeBlock(c1, c2, mat_eval, eta);
    FMCA::Matrix K00ref = KSig.block(c1.start_index(), c2.start_index(),
                                     c1.nsamplets(), c2.nsamplets());
    std::cout << "recursivelyComputeBlock error on top left corner: "
              << (K00.bottomRightCorner(c1.nsamplets(), c2.nsamplets()) -
                  K00ref)
                         .norm() /
                     K00ref.norm()
              << std::endl;
    std::cout << "case count: \n";
    for (int i = 0; i < 5; ++i)
      std::cout << "case " << i << ": "
                << FMCA::internal::recursivelyComputeBlockCounters[i]
                << std::endl;
  }
  {
    H2SampletTree &c1 = hst.sons(0).sons(1).sons(0).sons(1);
    H2SampletTree &c2 = hst.sons(0).sons(0).sons(1).sons(1).sons(0);
    FMCA::Matrix K00 =
        FMCA::internal::recursivelyComputeBlock(c1, c2, mat_eval, eta);
    FMCA::Matrix K00ref = KSig.block(c1.start_index(), c2.start_index(),
                                     c1.nsamplets(), c2.nsamplets());
    std::cout << "recursivelyComputeBlock error on top left corner: "
              << (K00.bottomRightCorner(c1.nsamplets(), c2.nsamplets()) -
                  K00ref)
                         .norm() /
                     K00ref.norm()
              << std::endl;
    std::cout << "case count: \n";
    for (int i = 0; i < 5; ++i)
      std::cout << "case " << i << ": "
                << FMCA::internal::recursivelyComputeBlockCounters[i]
                << std::endl;
  }
  // give compression error as indication
  FMCA::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, 1e5 * FMCA_ZERO_TOLERANCE);
  Scomp.compress(mat_eval);
  FMCA::SparseMatrix S(NPTS, NPTS);
  const auto &trips = Scomp.triplets();
  S.setFromTriplets(trips.begin(), trips.end());
  FMCA::SparseMatrix bla = S.selfadjointView<FMCA::Upper>();
  FMCA::Matrix Kcomp(bla);
  std::cout << "compression error: " << (KSig - Kcomp).norm() / KSig.norm()
            << std::endl;

  return 0;
}

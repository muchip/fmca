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

#define FMCA_CLUSTERSET_
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"

#define NPTS 300
#define DIM 2
#define MPOLE_DEG 3
#define LEAFSIZE 10

struct exponentialKernel {
  double operator()(const FMCA::Matrix &x, const FMCA::Matrix &y) const {
    return exp(-10 * (x - y).norm());
  }
};

using Interpolator = FMCA::TensorProductInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, exponentialKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  const double threshold = 1e-15;
  const Moments mom(P, MPOLE_DEG);
  const MatrixEvaluator mat_eval(mom, function);

  for (int dtilde = 1; dtilde <= 4; ++dtilde) {
    for (double eta = 1; eta >= 0; eta -= 0.5) {
      std::cout << "dtilde= " << dtilde << " eta= " << eta << std::endl;
      const SampletMoments samp_mom(P, dtilde - 1);
      H2SampletTree hst(mom, samp_mom, 0, P);
      T.tic();
      FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
      Scomp.init(hst, eta, 0);
      T.toc("planner:");

#if 0
      Scomp.compress(hst, hst, mat_eval, eta, threshold);
      const auto &trips = Scomp.pattern_triplets();
      Eigen::MatrixXd K;
      mat_eval.compute_dense_block(hst, hst, &K);
      hst.sampletTransformMatrix(K);
      // K.triangularView<Eigen::StrictlyLower>().setZero();
      Eigen::SparseMatrix<double> S(K.rows(), K.cols());
      S.setFromTriplets(trips.begin(), trips.end());
      std::cout << "compression error: " << (S - K).norm() / K.norm()
                << std::endl;
      std::cout << std::string(60, '-') << std::endl;
#endif
    }
  }
  return 0;
}

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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2
#define MPOLE_DEG 6

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar threshold = 1e-10;
  const FMCA::Index dtilde = 4;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);
  const Moments mom(P, mpole_deg);
  const usMatrixEvaluator mat_eval(mom, mom, function);
  const MatrixEvaluator smat_eval(mom, function);
  for (double eta = 1.2; eta >= 0.0; eta -= 0.2) {
    std::cout << "dtilde:                       " << dtilde << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    const SampletMoments samp_mom(P, dtilde - 1);
    H2SampletTree hst(mom, samp_mom, 0, P);
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
    T.tic();
    FMCA::internal::SampletMatrixCompressor<H2SampletTree> sScomp;
    sScomp.init(hst, eta, threshold);
    T.toc("symmetric planner:           ");
    T.tic();
    sScomp.compress(smat_eval);
    T.toc("symmetric compressor:        ");
    T.tic();
    const auto &strips = sScomp.triplets();
    T.toc("triplets:                    ");
    std::cout << "anz:                          "
              << std::round(strips.size() / FMCA::Scalar(NPTS)) << std::endl;
    // error computation
    FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
      y1 = col(FMCA::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
      x = hst.sampletTransform(x);
      y2.setZero();
      y1.setZero();
      for (const auto &i : strips) {
        y1(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col()) y1(i.col()) += i.value() * x(i.row());
      }
      for (const auto &i : trips) {
        y2(i.row()) += i.value() * x(i.col());
      }
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    err = sqrt(err / nrm);
    std::cout << "compression error:            " << err << std::endl
              << std::flush;
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

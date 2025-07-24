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
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 1000000
#define DIM 10

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1.);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  const FMCA::Scalar threshold = 1e-3;
  const FMCA::Scalar eta = 0.5;

  for (int dtilde = 2; dtilde <= 5; ++dtilde) {
    const FMCA::Index mpole_deg = 2 * (dtilde - 1);
    const Moments mom(P, mpole_deg);
    const MatrixEvaluator mat_eval(mom, function);
    std::cout << "dtilde:                       " << dtilde << std::endl;
    std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    const SampletMoments samp_mom(P, dtilde - 1);
    H2SampletTree hst(mom, samp_mom, 0, P);
    FMCA::clusterTreeStatistics(hst, P);
    T.tic();
    FMCA::internal::SampletMatrixCompressor<H2SampletTree, FMCA::CompareClusterStrict>
        Scomp;
    Scomp.init(hst, eta, 100 * FMCA_ZERO_TOLERANCE);
    T.toc("planner:                     ");
    T.tic();
    Scomp.compress(mat_eval);
    T.toc("compressor:                  ");
    T.tic();
    const auto &ap_trips = Scomp.triplets();
    std::cout << "anz (a-priori):               "
              << std::round(ap_trips.size() / FMCA::Scalar(NPTS)) << std::endl;
    T.toc("triplets:                    ");
#ifdef FMCA_VERBOSE
    FMCA::Index intervals = 17;
    FMCA::Vector values(intervals);
    values.setZero();
    for (auto i = 0; i < ap_trips.size(); ++i) {
      const FMCA::Scalar val =
          std::log(std::abs(ap_trips[i].value())) / std::log(10.);
      if (val >= 0)
        values[0] += 1;
      else if (val < -16)
        values[16] += 1;
      else
        values[FMCA::Index(-std::floor(val))] += 1.;
    }
    FMCA::Scalar bar_factor = 40 * ap_trips.size() / values.maxCoeff();
    bar_factor = bar_factor < FMCA_INF ? bar_factor : 0;
    for (auto i = 0; i < intervals; ++i) {
      std::cout << std::scientific << std::setprecision(2) << "1e-"
                << std::setw(2) << i << "|";
      std::cout << std::string(
                       std::ceil(bar_factor * values(i) / ap_trips.size()), '*')
                << std::endl;
    }
#endif
    T.tic();
    const auto &trips = Scomp.aposteriori_triplets_fast(threshold);
    std::cout << "anz (a-posteriori):           "
              << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;
#ifdef FMCA_VERBOSE
    values.setZero();
    for (auto i = 0; i < trips.size(); ++i) {
      const FMCA::Scalar val =
          std::log(std::abs(trips[i].value())) / std::log(10.);
      if (val >= 0)
        values[0] += 1;
      else if (val < -16)
        values[16] += 1;
      else
        values[FMCA::Index(-std::floor(val))] += 1.;
    }
    bar_factor = 40 * trips.size() / values.maxCoeff();
    bar_factor = bar_factor < FMCA_INF ? bar_factor : 0;
    for (auto i = 0; i < intervals; ++i) {
      std::cout << std::scientific << std::setprecision(2) << "1e-"
                << std::setw(2) << i << "|";
      std::cout << std::string(std::ceil(bar_factor * values(i) / trips.size()),
                               '*')
                << std::endl;
    }
#endif
    T.toc("triplets:                    ");
    FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      FMCA::Vector col = function.eval(P, P.col(hst.indices()[index]));
      y1 =
          col(Eigen::Map<const FMCA::iVector>(hst.indices(), hst.block_size()));
      x = hst.sampletTransform(x);
      y2.setZero();
      for (const auto &i : trips) {
        y2(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
      }
      y2 = hst.inverseSampletTransform(y2);
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

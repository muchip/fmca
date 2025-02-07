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
#include "../FMCA/src/Samplets/samplet_matrix_compressor_unsymmetric.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS_ROWS 200000
#define NPTS_COLS 100000
#define DIM 2

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
  const FMCA::CovarianceKernel function("MaternNu", 1., 1., .5);
  const FMCA::Matrix P_rows =
      0.5 * (FMCA::Matrix::Random(DIM, NPTS_ROWS).array() + 1);
  //   const FMCA::Matrix P_cols = P_rows;
  const FMCA::Matrix P_cols =
      0.5 * (FMCA::Matrix::Random(DIM, NPTS_COLS).array() + 1);

  const FMCA::Scalar threshold = 1e-9;
  const FMCA::Scalar eta = 0.5;

  for (int dtilde = 2; dtilde <= 6; ++dtilde) {
    const FMCA::Index mpole_deg = 2 * (dtilde - 1);
    const Moments mom_rows(P_rows, mpole_deg);
    const Moments mom_cols(P_cols, mpole_deg);
    const usMatrixEvaluator mat_eval(mom_rows, mom_cols, function);
    std::cout << "dtilde:                       " << dtilde << std::endl;
    std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    const SampletMoments samp_mom_rows(P_rows, dtilde - 1);
    const SampletMoments samp_mom_cols(P_cols, dtilde - 1);
    H2SampletTree hst_rows(mom_rows, samp_mom_rows, 0, P_rows);
    H2SampletTree hst_cols(mom_cols, samp_mom_cols, 0, P_cols);
    T.tic();
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_rows, hst_cols, eta, 1e-15);
    T.toc("planner:                     ");

    T.tic();
    Scomp.compress(mat_eval);
    T.toc("compressor:                  ");

    T.tic();
    const auto &ap_trips = Scomp.triplets();
    FMCA::Scalar ap_trips_time = T.toc();

    T.tic();
    const auto &trips = Scomp.aposteriori_triplets(threshold);
    FMCA::Scalar trips_time = T.toc();

    std::cout << "" << std::endl;
    std::cout << "anz (a-priori):               "
              << std::round(ap_trips.size() / FMCA::Scalar(NPTS_ROWS))
              << std::endl;
    std::cout << "anz (a-posteriori):           "
              << std::round(trips.size() / FMCA::Scalar(NPTS_ROWS))
              << std::endl;
    std::cout << "" << std::endl;
    std::cout << "time (a-priori):              " << ap_trips_time << "sec." << std::endl;
    std::cout << "time (a-posteriori):          " << trips_time << "sec." << std::endl;
    std::cout << "" << std::endl;
    // T.toc("triplets:                    ");

    // error computation
    FMCA::Vector x(NPTS_COLS), y1(NPTS_ROWS), y2_apriori(NPTS_ROWS),
        y2_aposteriori(NPTS_ROWS);
    FMCA::Scalar err_apriori = 0;
    FMCA::Scalar nrm_apriori = 0;
    FMCA::Scalar err_aposteriori = 0;
    FMCA::Scalar nrm_aposteriori = 0;

    for (auto i = 0; i < 100; ++i) {
      FMCA::Index index = rand() % NPTS_COLS;
      x.setZero();
      x(index) = 1;

      FMCA::Vector col =
          function.eval(P_rows, P_cols.col(hst_cols.indices()[index]));
      y1 = col(Eigen::Map<const FMCA::iVector>(hst_rows.indices(),
                                               hst_rows.block_size()));
      x = hst_cols.sampletTransform(x);
      y2_apriori.setZero();
      y2_aposteriori.setZero();

      for (const auto &triplet : ap_trips) {
        y2_apriori(triplet.row()) += triplet.value() * x(triplet.col());
      }
      y2_apriori = hst_rows.inverseSampletTransform(y2_apriori);
      err_apriori += (y1 - y2_apriori).squaredNorm();
      nrm_apriori += y1.squaredNorm();

      for (const auto &triplet : trips) {
        y2_aposteriori(triplet.row()) += triplet.value() * x(triplet.col());
      }
      y2_aposteriori = hst_rows.inverseSampletTransform(y2_aposteriori);
      err_aposteriori += (y1 - y2_aposteriori).squaredNorm();
      nrm_aposteriori += y1.squaredNorm();
    }
    std::cout << "Compression error unsymmetric a priori          "
              << sqrt(err_apriori / nrm_apriori) << std::endl
              << std::flush;
    std::cout << "Compression error unsymmetric a posteriori      "
              << sqrt(err_aposteriori / nrm_aposteriori) << std::endl
              << std::flush;
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

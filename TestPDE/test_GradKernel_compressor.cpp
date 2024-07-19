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

#include <iostream>

#include "../FMCA/GradKernel"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"
#include "read_files_txt.h"

#define DIM 2
#define MPOLE_DEG 6

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
// using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::GradKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::GradKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::GradKernel function("MATERN32", 1, 1, 0);
  FMCA::Matrix P_sources;
  FMCA::Matrix P_quad;
  readTXT("data/vertices_square_01.txt", P_sources, 2);
  readTXT("data/quadrature7_points_square_01.txt", P_quad, 2);
  int NPTS_SOURCE = P_sources.cols();
  int NPTS_QUAD = P_quad.cols();
  // const FMCA::Matrix P_sources = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_SOURCE).array() + 1);
  // const FMCA::Matrix P_quad = 0.5 * (FMCA::Matrix::Random(DIM, NPTS_QUAD).array() + 1);
  const FMCA::Scalar threshold = 1e-6;
  const FMCA::Index dtilde = 4;
  const Moments mom_sources(P_sources, MPOLE_DEG);
  const Moments mom_quad(P_quad, MPOLE_DEG);
  const usMatrixEvaluator mat_eval(mom_sources, mom_quad, function);

  for (double eta = 1; eta >= 0.4; eta -= 0.2) {
    std::cout << "dtilde:                       " << dtilde << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    const SampletMoments samp_mom_sources(P_sources, dtilde - 1);
    const SampletMoments samp_mom_quad(P_quad, dtilde - 1);
    H2SampletTree hst_sources(mom_sources, samp_mom_sources , 0, P_sources);
    H2SampletTree hst_quad(mom_quad, samp_mom_quad, 0, P_quad);

    T.tic();
    FMCA::internal::SampletMatrixCompressorUnsymmetric<H2SampletTree> Scomp;
    Scomp.init(hst_sources, hst_quad, eta, threshold);
    T.toc("unsymmetric planner:         ");
    T.tic();
    Scomp.compress(mat_eval);
    T.toc("unsymmetric compressor:      ");
    T.tic();
    const auto &trips = Scomp.triplets();
    T.toc("triplets:                    ");
    std::cout << "anz:                          "
              << std::round(trips.size() / FMCA::Scalar(NPTS_SOURCE)) << std::endl;

    // error computation
    FMCA::Vector x(NPTS_QUAD), y1(NPTS_SOURCE), y2(NPTS_SOURCE);
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % NPTS_QUAD;
      x.setZero();
      x(index) = 1;
      FMCA::Vector col = function.eval(P_sources, P_quad.col(hst_quad.indices()[index]));
      y1 =
          col(Eigen::Map<const FMCA::iVector>(hst_sources.indices(), hst_sources.block_size()));
      x = hst_quad.sampletTransform(x);
      y2.setZero();
      for (const auto &i : trips) {
        y2(i.row()) += i.value() * x(i.col());
      }
      y2 = hst_sources.inverseSampletTransform(y2);
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


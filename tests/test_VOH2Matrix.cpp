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
#include "../FMCA/H2Matrix"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 2
#define MPOLE_DEG 3

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::VariableOrderNystromMoments<Interpolator>;
using MatrixEvaluator =
    FMCA::VariableOrderNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::VOH2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  FMCA::Index maxlevel = 0;
  FMCA::ClusterTree ct(P, 4);
  for (const auto &node : ct)
    maxlevel = maxlevel > node.level() ? maxlevel : node.level();
  std::cout << "maximum tree level: " << maxlevel << std::endl;
  FMCA::iVector degs(maxlevel + 1);
  for (FMCA::Index i = 0; i <= maxlevel; ++i)
    degs(i) = 1 + std::floor(FMCA::Scalar(maxlevel - i) / DIM);
  const Moments mom(P, degs);
  T.tic();
  H2ClusterTree hct(mom, ct);
  T.toc("H2 cluster tree:");
  const MatrixEvaluator mat_eval(mom, function);
  for (FMCA::Scalar eta = 0.5; eta >= 0.5; eta *= 0.5) {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    const H2Matrix hmat(hct, mat_eval, eta);
    T.toc("elapsed time:                ");
    hmat.statistics();
    {
      FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
      FMCA::Scalar err = 0;
      FMCA::Scalar nrm = 0;
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        FMCA::Vector col = function.eval(P, P.col(ct.indices()[index]));
        y1 =
            col(Eigen::Map<const FMCA::iVector>(ct.indices(), ct.block_size()));
        y2 = hmat * x;
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error:            " << err << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }
  }
  return 0;
}

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

#define NPTS 500000
#define DIM 3

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
  const FMCA::ClusterTree ct(P, 2);
  const FMCA::Scalar eta = 0.5;

  FMCA::Index maxlevel = 0;
  for (const auto &node : ct)
    maxlevel = maxlevel > node.level() ? maxlevel : node.level();
  std::cout << "eta:                          " << eta << std::endl;
  std::cout << "maximum tree level:           " << maxlevel << std::endl;

  FMCA::iVector degs(maxlevel + 1);
  for (FMCA::Index i = 0; i <= maxlevel; ++i)
    degs(i) = std::floor(FMCA::Scalar(maxlevel - i) / DIM);

  const Moments mom(P, degs);
  T.tic();
  const H2ClusterTree hct(mom, ct);
  T.toc("H2 cluster tree:             ");
  const MatrixEvaluator mat_eval(mom, function);
  T.tic();
  const H2Matrix hmat(hct, mat_eval, eta);
  T.toc("H2 matrix:                   ");
  hmat.statistics();
  {
    FMCA::Matrix X(NPTS, 10), Y1(NPTS, 10), Y2(NPTS, 10);
    X.setZero();
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      X(index, i) = 1;
      FMCA::Vector col = function.eval(P, P.col(ct.indices()[index]));
      Y1.col(i) =
          col(Eigen::Map<const FMCA::iVector>(ct.indices(), ct.block_size()));
    }
    Y2 = hmat * X;
    const FMCA::Scalar err = (Y1 - Y2).squaredNorm();
    const FMCA::Scalar nrm = Y1.squaredNorm();
    std::cout << "compression error:            " << sqrt(err / nrm)
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  return 0;
}

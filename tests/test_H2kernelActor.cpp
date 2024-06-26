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

#define NPTS 10000
#define DIM 2
#define MPOLE_DEG 4

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using MatrixEvaluatorUS =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1);
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  const Moments mom(P, MPOLE_DEG);
  FMCA::Tictoc T;
  for (FMCA::Scalar eta = 0.8; eta >= 0.1; eta *= 0.5) {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    const FMCA::H2kernelActor<FMCA::CovarianceKernel, Moments, H2ClusterTree,
                              MatrixEvaluatorUS>
        hact(function, P, P, MPOLE_DEG, eta);
    T.toc("init action: ");
    {
      FMCA::Vector x(NPTS), y3(NPTS);
      FMCA::Scalar err2 = 0;
      FMCA::Scalar nrm2 = 0;
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        FMCA::Vector col = function.eval(P, P.col(index));
        T.tic();
        y3 = hact.action(x);
        T.toc("performing action: ");
        err2 += (y3 - col).squaredNorm();
        nrm2 += col.squaredNorm();
      }
      err2 = sqrt(err2 / nrm2);
      std::cout << "action error:                 " << err2 << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }
  }
  return 0;
}

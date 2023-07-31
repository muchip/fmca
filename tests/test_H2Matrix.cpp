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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define NPTS 100000
#define DIM 100
#define MPOLE_DEG 5

using Interpolator = FMCA::WeightedTotalDegreeInterpolator;
using Moments = FMCA::WeightedNystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::CovarianceKernel function("EXPONENTIAL", 1);
  FMCA::Matrix P(DIM, NPTS);
  FMCA::Vector b(DIM);
  std::vector<FMCA::Scalar> w(DIM);
  for (FMCA::Index i = 0; i < DIM; ++i) {
    b(i) = 1. / FMCA::Scalar(i + 1) / FMCA::Scalar(i + 1);
    w[i] = log(2. / b(i) + sqrt(1 + 4. / b(i) / b(i)));
  }
  std::cout << "w: ";
  for (FMCA::Index i = 0; i < DIM; ++i) {
    w[i] /= w[0];
    std::cout << w[i] << " ";
  }
  std::cout << b.transpose() << std::endl;
  P.setRandom();
  P = b.asDiagonal() * P;
  FMCA::Index cur_n_pts = 0;
  const Moments mom(P, w, MPOLE_DEG);

  H2ClusterTree ct(mom, 0, P);
  std::vector<FMCA::Matrix> bbs;
  for (const auto &node : ct) {
    if (!node.nSons()) bbs.push_back(node.bb());
  }
  //FMCA::IO::plotBoxes("boxes.vtk", bbs);
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ct, mom);
  const MatrixEvaluator mat_eval(mom, function);
  for (FMCA::Scalar eta = 0.8; eta >= 0.2; eta *= 0.5) {
    std::cout << "eta:                          " << eta << std::endl;
    T.tic();
    const H2Matrix hmat(ct, mat_eval, eta);
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
        y1 = col(ct.indices());
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

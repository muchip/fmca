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

#include "../FMCA/H2Matrix"

#define NPTS 20000
#define DIM 2
#define MPOLE_DEG 3
#define LEAFSIZE 10

template <typename Functor>
FMCA::Vector matrixColumnGetter(const FMCA::Matrix &P,
                                const std::vector<FMCA::Index> &idcs,
                                const Functor &fun, FMCA::Index colID) {
  FMCA::Vector retval(P.cols());
  retval.setZero();
  for (auto i = 0; i < retval.size(); ++i)
    retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
  return retval;
}

struct exponentialKernel {
  double operator()(const FMCA::Matrix &x, const FMCA::Matrix &y) const {
    return exp(-10 * (x - y).norm());
  }
};

using Interpolator = FMCA::TensorProductInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, exponentialKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main() {
  const auto function = exponentialKernel();
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  const Moments mom(P, MPOLE_DEG);
  H2ClusterTree ct(mom, 0, P);
  FMCA::internal::compute_cluster_bases_impl::check_transfer_matrices(ct, mom);
  const MatrixEvaluator mat_eval(mom, function);
  for (FMCA::Scalar eta = 0.8; eta >= 0.1; eta *= 0.5) {
    std::cout << "eta= " << eta << std::endl;
    const H2Matrix hmat(ct, mat_eval, eta);
    hmat.get_statistics();
    {
      FMCA::Vector x(NPTS), y1(NPTS), y2(NPTS);
      FMCA::Scalar err = 0;
      FMCA::Scalar nrm = 0;
      for (auto i = 0; i < 10; ++i) {
        FMCA::Index index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = matrixColumnGetter(P, ct.indices(), function, index);
        y2 = hmat * x;
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }
  }
  return 0;
}

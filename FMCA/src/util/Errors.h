// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_ERRORS_H_
#define FMCA_UTIL_ERRORS_H_

#include <vector>

#include "Macros.h"
#include "SparseMatrix.h"

namespace FMCA {

template <typename Functor>
Eigen::VectorXd matrixMultiplier(const Eigen::MatrixXd &P,
                                 const std::vector<FMCA::IndexType> &idcs,
                                 const Functor &fun, const Eigen::VectorXd &x) {
  Eigen::VectorXd retval(x.size());
  retval.setZero();
  for (auto i = 0; i < x.size(); ++i)
    for (auto j = 0; j < x.size(); ++j)
      retval(i) += fun(P.col(idcs[i]), P.col(idcs[j])) * x(j);
  return retval;
}
////////////////////////////////////////////////////////////////////////////////
template <typename Functor>
Eigen::VectorXd matrixColumnGetter(const Eigen::MatrixXd &P,
                                   const std::vector<FMCA::IndexType> &idcs,
                                   const Functor &fun, Eigen::Index colID) {
  Eigen::VectorXd retval(P.cols());
  retval.setZero();
  for (auto i = 0; i < retval.size(); ++i)
    retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
  return retval;
}

template <typename Functor, typename Derived>
double errorEstimatorSymmetricCompressor(
    const std::vector<Eigen::Triplet<double>> &trips, const Functor &function,
    const FMCA::SampletTreeBase<Derived> &hst, const Eigen::MatrixXd &P) {
  unsigned int npts = P.cols();
  Eigen::VectorXd x(npts), y1(npts), y2(npts);
  double err = 0;
  double nrm = 0;
  for (auto i = 0; i < 100; ++i) {
    unsigned int index = rand() % P.cols();
    x.setZero();
    x(index) = 1;
    y1 = FMCA::matrixColumnGetter(P, hst.indices(), function, index);
    x = hst.sampletTransform(x);
    y2 = FMCA::SparseMatrix<double>::symTripletsTimesVector(trips, x);
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  return sqrt(err / nrm);
}

} // namespace FMCA

#endif

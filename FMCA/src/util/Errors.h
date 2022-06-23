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
Vector matrixMultiplier(const Matrix &P, const std::vector<Index> &idcs,
                        const Functor &fun, const Vector &x) {
  Vector retval(x.size());
  retval.setZero();
  for (auto i = 0; i < x.size(); ++i)
    for (auto j = 0; j < x.size(); ++j)
      retval(i) += fun(P.col(idcs[i]), P.col(idcs[j])) * x(j);
  return retval;
}
////////////////////////////////////////////////////////////////////////////////
template <typename Functor>
Vector matrixColumnGetter(const Matrix &P, const std::vector<Index> &idcs,
                          const Functor &fun, Index colID) {
  Vector retval(P.cols());
  retval.setZero();
  for (auto i = 0; i < retval.size(); ++i)
    retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
  return retval;
}

template <typename Functor, typename Derived>
Scalar errorEstimatorSymmetricCompressor(
    const std::vector<Eigen::Triplet<Scalar>> &trips, const Functor &function,
    const FMCA::SampletTreeBase<Derived> &hst, const Matrix &P) {
  Index npts = P.cols();
  Vector x(npts), y1(npts), y2(npts);
  Scalar err = 0;
  Scalar nrm = 0;
  for (auto i = 0; i < 5; ++i) {
    Index index = rand() % P.cols();
    x.setZero();
    x(index) = 1;
    y1 = FMCA::matrixColumnGetter(P, hst.indices(), function, index);
    x = hst.sampletTransform(x);
    y2 = FMCA::SparseMatrix<Scalar>::symTripletsTimesVector(trips, x);
    y2 = hst.inverseSampletTransform(y2);
    err += (y1 - y2).squaredNorm();
    nrm += y1.squaredNorm();
  }
  return sqrt(err / nrm);
}
} // namespace FMCA

#endif

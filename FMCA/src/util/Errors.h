// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_ERRORS_H_
#define FMCA_UTIL_ERRORS_H_

#include "Macros.h"
#include <vector>

namespace FMCA {

template <typename Functor>
Vector matrixMultiplier(const Matrix &P,
                                 const std::vector<Index> &idcs,
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
Vector matrixColumnGetter(const Matrix &P,
                                   const std::vector<Index> &idcs,
                                   const Functor &fun, Eigen::Index colID) {
  Vector retval(P.cols());
  retval.setZero();
  for (auto i = 0; i < retval.size(); ++i)
    retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
  return retval;
}

}  // namespace FMCA

#endif

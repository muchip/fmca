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
#ifndef FMCA_UTIL_BINOMIALCOEFFICIENT_H_
#define FMCA_UTIL_BINOMIALCOEFFICIENT_H_

#include <array>

#include "Macros.h"

namespace FMCA {

IndexType binomialCoefficient(IndexType n, IndexType k) {
  if (k > n)
    return 0;
  else if (n == k)
    return 1;
  else
    return binomialCoefficient(n - 1, k - 1) + binomialCoefficient(n - 1, k);
}

template <IndexType Dim>
IndexType multinomialCoefficient(const std::array<IndexType, Dim> &alpha,
                                 const std::array<IndexType, Dim> &beta) {
  IndexType retval = 1;
  for (auto i = 0; i < Dim; ++i)
    retval *= binomialCoefficient(alpha[i], beta[i]);
  return retval;
}
} // namespace FMCA

#endif

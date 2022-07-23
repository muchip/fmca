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
#ifndef FMCA_UTIL_BINOMIALCOEFFICIENT_H_
#define FMCA_UTIL_BINOMIALCOEFFICIENT_H_

#include "Macros.h"

namespace FMCA {

Index binomialCoefficient(Index n, Index k) {
  if (k > n)
    return 0;
  else if (n == k)
    return 1;
  else
    return binomialCoefficient(n - 1, k - 1) + binomialCoefficient(n - 1, k);
}

template <typename MultiIndex>
Index multinomialCoefficient(const MultiIndex &alpha, const MultiIndex &beta) {
  Index retval = 1;
  for (auto i = 0; i < alpha.size(); ++i)
    retval *= binomialCoefficient(alpha[i], beta[i]);
  return retval;
}
} // namespace FMCA

#endif

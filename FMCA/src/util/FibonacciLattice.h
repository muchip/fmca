// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_FIBONACCILATTICE_H_
#define FMCA_UTIL_FIBONACCILATTICE_H_

#include "Macros.h"

namespace FMCA {

Matrix FibonacciLattice(const Index N) {
  Matrix retval(3, N);
  const Scalar golden_angle = FMCA_PI * (3.0 - std::sqrt(5.0));
  for (Index i = 0; i < N; ++i) {
    const Scalar z = 1.0 - (2.0 * i + 1.0) / N;
    const Scalar radius = std::sqrt(1.0 - z * z);
    const Scalar phi = golden_angle * i;
    const Scalar x = radius * std::cos(phi);
    const Scalar y = radius * std::sin(phi);
    retval.col(i) << x, y, z;
  }
  return retval;
}

}  // namespace FMCA

#endif

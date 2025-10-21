// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_UNIFORMSPHERICALPOINTS_H_
#define FMCA_UTIL_UNIFORMSPHERICALPOINTS_H_

#include "NormalDistribution.h"

namespace FMCA {

Matrix uniformSphericalPoints(Index N, size_t seed = 0) {
  NormalDistribution nd(0, 1, seed);
  Matrix P = nd.randN(3, N);
  for (Index i = 0; i < P.cols(); ++i) P.col(i).normalize();
  return P;
}

}  // namespace FMCA

#endif

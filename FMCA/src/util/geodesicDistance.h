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
///
#ifndef FMCA_UTIL_GEODESICDISTANCE_H_
#define FMCA_UTIL_GEODESICDISTANCE_H_

#include "Macros.h"

namespace FMCA {
Scalar geodesicDistance(const Vector &a, const Vector &b) {
  const Scalar dot = a.dot(b);
  const Scalar clamped_dot = std::min(1., std::max(-1., dot));
  return std::acos(clamped_dot);
}

}  // namespace FMCA
#endif

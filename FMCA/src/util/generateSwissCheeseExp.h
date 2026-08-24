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
#ifndef FMCA_UTIL_GENERATESWISSCHEESE_H_
#define FMCA_UTIL_GENERATESWISSCHEESE_H_

#include "Macros.h"

namespace FMCA {
Matrix generateSwissCheeseExp(Index dim, Index npts) {
  Index nholes = 0;
  Scalar min_rad = 0;
  Scalar max_rad = 0;
  std::cout << "using swiss cheese exp\n";
  switch (dim) {
    case 1:
      nholes = 100;
      min_rad = 0.004;
      max_rad = 0.006;
      break;
    case 2:
      nholes = 3000;
      min_rad = 0.009;
      max_rad = 0.01;
      break;
    case 3:
      nholes = 10000;
      min_rad = 0.03;
      max_rad = 0.034;
      break;
    default:
      return Matrix(0, 0);
  }

  srand(0);
  Matrix retval(dim, npts);
  retval.setZero();
  // generate holes
  std::vector<std::pair<Vector, Scalar>> holes;
  for (Index i = 0; i < nholes; ++i) {
    holes.push_back(std::make_pair(
        1.1 * 0.5 * (Eigen::VectorXd::Random(dim).array() + 1),
        min_rad + (min_rad - max_rad) * log(1. - 1. * rand() / RAND_MAX)));
  }
  for (Index i = 0; i < npts; ++i) {
    Vector cur_pt;
    bool found_pt = false;
    while (!found_pt) {
      cur_pt = Eigen::VectorXd::Random(dim).array();
      Scalar nrm = cur_pt.norm();
      if (nrm <= 1) {
        cur_pt = -cur_pt / cur_pt.norm() * log(1. - 1. * rand() / RAND_MAX);
      } else
        continue;
      if (cur_pt.maxCoeff() > 1 || cur_pt.minCoeff() < 0) continue;
      bool hit_hole = false;
      for (Index j = 0; j < nholes; ++j)
        if ((cur_pt - holes[j].first).norm() < holes[j].second) {
          hit_hole = true;
          break;
        }
      if (!hit_hole) found_pt = true;
    }
    retval.col(i) = cur_pt;
  }

  return retval;
}
}  // namespace FMCA
#endif

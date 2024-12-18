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

#include "../FMCA/src/util/CombiIndexSet.h"
#include "../FMCA/src/util/MultiIndexSet.h"

#define DIM 2

int main() {
  FMCA::MultiIndexSet set(3, 4);
  FMCA::CombiIndexSet cset(3, 4);
  std::vector<FMCA::Scalar> w(3);
  w[0] = 1;
  w[1] = 2;
  w[2] = 3;
  FMCA::CombiIndexSet<FMCA::WeightedTotalDegree> wcset(3, 4, w);

  for (const auto &it : set.index_set()) {
    for (const auto &it2 : it) std::cout << it2 << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (const auto &it : cset.index_set()) {
    for (const auto &it2 : it.first) std::cout << it2 << " ";
    std::cout << " cw= " << it.second << std::endl;
  }
  std::cout << std::endl;

  for (const auto &it : wcset.index_set()) {
    for (const auto &it2 : it.first) std::cout << it2 << " ";
    std::cout << " cw= " << it.second << std::endl;
  }
  std::cout << std::endl;
  return 0;
}

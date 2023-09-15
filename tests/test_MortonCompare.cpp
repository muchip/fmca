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

#include "../FMCA/Clustering"
#include "../FMCA/src/Clustering/MortonCompare.h"
#include "../FMCA/src/util/IO.h"

#define DIM 3
#define NPTS 10000

int main() {
  using namespace FMCA;
  FMCA::Matrix pts(2, 1024);
  FMCA::iVector idcs(1024);
  idcs = FMCA::iVector::LinSpaced(1024, 0, 1023);
  FMCA::iVector iidcs = FMCA::iVector::LinSpaced(32, 0, 31);

  // std::cout << idcs << std::endl;
  FMCA::Index k = 0;
  for (FMCA::Index i = 0; i < 32; ++i) {
    for (FMCA::Index j = 0; j < 32; ++j) {
      pts.col(k) << iidcs(i), iidcs(j);
      ++k;
    }
  }

  std::sort(idcs.begin(), idcs.end(), MortonCompare<FMCA::Matrix>(pts));
  // std::cout << "------" << std::endl;
  // std::cout << idcs << std::endl;
  FMCA::IO::print2m("output.m", "P", pts, "w");
  FMCA::IO::print2m("output.m", "I", idcs, "a");
  return 0;
}

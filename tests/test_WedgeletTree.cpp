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

#include "../FMCA/Wedgelets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 3
#define NPTS 10000

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = FMCA::IO::ascii2Matrix("P.dat");
  const FMCA::Matrix rgb = FMCA::IO::ascii2Matrix("rgb.dat");
  std::cout << rgb.topRows(10) << std::endl << "......." << std::endl;
  FMCA::WedgeletTree<double> wt(P, 5);
  wt.computeWedges(P, rgb, 3, 1e-1);
  FMCA::Vector hits(P.cols());
  hits.setZero();
  for (const auto &it : wt)
    if (!it.nSons() && it.block_size()) {
      for (FMCA::Index j = 0; j < it.block_size(); ++j) {
        assert(hits(it.indices()[j]) == 0 && "duplicate index");
        hits(it.indices()[j]) = 1;
      }
    }
  assert(hits.sum() == hits.size() && "missing index");
  return 0;
}

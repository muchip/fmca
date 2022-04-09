// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/src/util/MultiIndexSet.h"
#include "../FMCA/src/util/Tictoc.h"
#include "TestParameters.h"

int main() {
  FMCA::Tictoc T;
  T.tic();
  FMCA::MultiIndexSet<FMCA::TensorProduct> tp(4, 3);
  T.toc("tp: ");
  T.tic();
  FMCA::MultiIndexSet<FMCA::TotalDegree> td(6, 40);
  T.toc("td: ");
  T.tic();
  FMCA::MultiIndexSet<FMCA::WeightedTotalDegree> wtd(
      6, 40, std::vector<double>({1., 1., 1., 1., 1., 1.}));
  T.toc("wtd: ");
  assert(td.index_set() == wtd.index_set() && "mismatch");
#if 0
  for (auto &&it : tp.index_set()) {
    for (auto &&it2 : it)
      std::cout << it2 << " ";
    std::cout << std::endl;
  }
  std::cout << "\n";

  for (auto &&it : td.index_set()) {
    for (auto &&it2 : it)
      std::cout << it2 << " ";
    std::cout << std::endl;
  }
  std::cout << "\n";

  for (auto &&it : wtd.index_set()) {
    for (auto &&it2 : it)
      std::cout << it2 << " ";
    std::cout << std::endl;
  }
  std::cout << "\n";
#endif
  return 0;
}

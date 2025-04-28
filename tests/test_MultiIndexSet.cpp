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
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2

int main() {
  FMCA::Tictoc T;
  FMCA::MultiIndexSet set(3, 4);
  FMCA::CombiIndexSet cset(3, 4);
  FMCA::Index dim = 200;
  std::vector<FMCA::Scalar> w(dim);
  for (FMCA::Index i = 0; i < dim; ++i) w[i] = (i + 1);
  T.tic();
  FMCA::CombiIndexSet<FMCA::WeightedTotalDegree> wcset(dim, 35, w);
  std::cout << wcset.index_set().size() << std::endl;
  T.toc("bla1: ");
  T.tic();
  FMCA::CombiIndexSet<FMCA::WeightedDroplet> wcset2(dim, 35, w);
  std::cout << wcset2.index_set().size() << std::endl;
  assert(wcset2.index_set().size() == wcset.index_set().size());
  T.toc("bla2: ");
  auto it1 = wcset.index_set().begin();
  auto it2 = wcset.index_set().begin();
  while (it1 != wcset.index_set().end()) {
    assert(it1->first == it2->first && it1->second == it2->second);
    ++it1;
    ++it2;
  }
  return 0;

  for (const auto &it : wcset.index_set()) {
    for (const auto &it2 : it.first) std::cout << it2 << " ";
    std::cout << " cw= " << it.second << std::endl;
  }
  std::cout << "----------" << std::endl;
  for (const auto &it : wcset2.index_set()) {
    for (const auto &it2 : it.first) std::cout << it2 << " ";
    std::cout << " cw= " << it.second << std::endl;
  }

  return 0;
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

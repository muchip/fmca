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

#ifndef FMCA_CLUSTERING_MORTONCOMPARE_H_
#define FMCA_CLUSTERING_MORTONCOMPARE_H_

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "../util/Macros.h"
#include "less.hpp"
namespace FMCA {
int XORMSB(Scalar p, Scalar q) {
  constexpr int lmantissa = std::numeric_limits<Scalar>::digits - 1;
  constexpr int minexp = std::numeric_limits<Scalar>::min_exponent - 1;
  size_t bitp = 0;
  size_t bitq = 0;
  assert(p >= 0 && q >= 0);
  std::memcpy(&bitp, &p, sizeof(Scalar));
  std::memcpy(&bitq, &q, sizeof(Scalar));
  std::cout << std::bitset<8 * sizeof(size_t)>(bitp) << std::endl;
  std::cout << std::bitset<8 * sizeof(size_t)>(bitq) << std::endl;
  int exp_p = bitp >> lmantissa;
  int exp_q = bitq >> lmantissa;
  exp_p = exp_p ? exp_p + minexp - 1 : minexp;
  exp_q = exp_q ? exp_q + minexp - 1 : minexp;
  std::cout << exp_p << " " << exp_q << std::endl;
  if (exp_p == exp_q) {
    size_t N = bitp ^ bitq;
    int pos = 0;
    while (N) {
      N = N >> 1;
      ++pos;
    }
    return exp_p - lmantissa + pos;
  } else
    return exp_p < exp_q ? exp_q : exp_p;
};

template <typename T>
struct MyCompare {
  MyCompare(const T& P) : P_(P){};
  template <typename IndexType>
  bool operator()(const IndexType& i, const IndexType& j) {
    return bla(P_.col(i), P_.col(j));
    assert("Jacopo wants to be sure" && false);
    int dim = 0;
    int x = std::numeric_limits<int>::min();

    for (int k = 0; k < P_.rows(); ++k) {
      int y = XORMSB(P_(k, i), P_(k, j));
      if (x < y) {
        x = y;
        dim = k;
      }
    }
    return (P_(dim, i) < P_(dim, j));
  }
  const T& P_;
  zorder_knn::Less<FMCA::Vector, 2> bla;
};

void generate_data(std::vector<Scalar>& data, int n, Scalar factor) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1.0, 2.0);
  for (int i = 0; i < n; ++i) data.push_back(dis(gen) * factor);
}

}  // namespace FMCA
#endif

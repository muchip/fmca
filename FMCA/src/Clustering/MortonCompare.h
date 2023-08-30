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

#include "../util/Macros.h"

namespace FMCA {

template <typename T>
struct MortonCompare;

namespace internal {
/**
 *  \brief computes the most significant bit difference
 *
 *  the implementation of the following is based on STANN see [Connor and Kumar:
 *  Fast construction of k-nearest neighbor graphs for point clouds, IEEE
 *  Transactions on Visualization and Computer Graphics, 16(4):599-608, 2010]
 *  Instead of bit-shifts, we employ here stl functionality to obtain the
 *  mantissa in integer format. In particular, size_t should always have more
 *  bits than the mantissa of a float (unless non trivial types are used, e.g.
 *  long double)
 **/
int mostSignificantBitDifference(Scalar p, Scalar q) {
  constexpr int digits = std::numeric_limits<Scalar>::digits;
  if (p == q) return std::numeric_limits<int>::min();
  int exp_p = 0;
  int exp_q = 0;
  const size_t mant_p = std::scalbn(std::frexp(std::abs(p), &exp_p), digits);
  const size_t mant_q = std::scalbn(std::frexp(std::abs(q), &exp_q), digits);
  if (exp_p == exp_q) {
    size_t N = mant_p ^ mant_q;
    int pos = 0;
    while (N) {
      N = N >> 1;
      ++pos;
    }
    pos -= digits;
    return exp_p + pos;
  } else
    return exp_p < exp_q ? exp_q : exp_p;
}
}  // namespace internal

/**
 *  \brief custom comparison struct to generate Morton ordering in d dimensions
 *
 *  the implementation of the following is based on STANN see [Connor and Kumar:
 *  Fast construction of k-nearest neighbor graphs for point clouds, IEEE
 *  Transactions on Visualization and Computer Graphics, 16(4):599-608, 2010]
 **/
template <>
struct MortonCompare<iMatrix> {
  MortonCompare(const iMatrix& P) : P_(P){};
  template <typename IndexType>
  bool operator()(const IndexType& i, const IndexType& j) {
    int ind = 0;
    int x = std::numeric_limits<int>::min();
    for (int k = P_.rows() - 1; k >= 0; --k) {
      int y = P_(k, i) ^ P_(k, j);
      if ((x < y) && (x < (x ^ y))) {
        x = y;
        ind = k;
      }
    }
    return (P_(ind, i) < P_(ind, j));
  }
  const iMatrix& P_;
};

/**
 *  \brief custom comparison struct to generate Morton ordering in d dimensions
 *
 *  the implementation of the following is based on STANN see [Connor and Kumar:
 *  Fast construction of k-nearest neighbor graphs for point clouds, IEEE
 *  Transactions on Visualization and Computer Graphics, 16(4):599-608, 2010]
 **/
template <>
struct MortonCompare<Matrix> {
  MortonCompare(const Matrix& P) : P_(P){};
  template <typename IndexType>
  bool operator()(const IndexType& i, const IndexType& j) {
    int ind = 0;
    int x = std::numeric_limits<int>::min();
    for (int k = P_.rows() - 1; k >= 0; --k) {
      if ((P_(k, i) < 0) != (P_(k, j) < 0)) return P_(k, i) < P_(k, j);
      int y = internal::mostSignificantBitDifference(P_(k, i), P_(k, j));
      if (x < y) {
        x = y;
        ind = k;
      }
    }
    return (P_(ind, i) < P_(ind, j));
  }
  const Matrix& P_;
};

}  // namespace FMCA
#endif

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
#ifndef FMCA_UTIL_DROPLET_H_
#define FMCA_UTIL_DROPLET_H_

#include "Macros.h"

namespace FMCA {

/**
 *  \brief enumerates all indices which are entrywise smaller or equal than
 *         a given multi index n;
 **/
template <typename MultiIndexSet>
void tensorProductDroplet(const typename MultiIndexSet::key_type &n,
                          MultiIndexSet &set) {
  const Index dim = n.size();
  typename MultiIndexSet::key_type mult_ind(dim);
  Index p = 0;

  set.clear();
  std::fill(mult_ind.begin(), mult_ind.end(), 0);
  while (mult_ind[dim - 1] <= n[dim - 1]) {
    if (mult_ind[p] > n[p]) {
      mult_ind[p] = 0;
      ++p;
    } else {
      set.insert(mult_ind);
      p = 0;
    }
    mult_ind[p] += 1;
  }

  return;
}

/**
 *  \brief enumerates all indices whose w weighted ell1 norm is smaller
 *         than q
 **/
template <typename VectorT, typename MultiIndexSet>
void weightedTotalDegreeDroplet(const VectorT &w, const Scalar q,
                                MultiIndexSet &set) {
  const Index dim = w.size();
  typename MultiIndexSet::key_type mult_ind(dim);
  Scalar scap = 0;
  Index p = 0;

  set.clear();
  std::fill(mult_ind.begin(), mult_ind.end(), 0);
  while (mult_ind[dim - 1] * w[dim - 1] <= q) {
    if (scap > q) {
      scap -= mult_ind[p] * w[p];
      mult_ind[p] = 0;
      ++p;
    } else {
      set.insert(mult_ind);
      p = 0;
    }
    mult_ind[p] += 1;
    scap += w[p];
  }

  return;
}

/**
 *  \brief computes the combination weight using a droplet algorithm
 **/
template <typename VectorT>
int weightedTotalDegreecombiWeightDroplet(const VectorT &w, const Scalar q) {
  int cw = 0;
  const Index dim = w.size();
  std::vector<Index> mult_ind(dim, 0);
  Scalar scap = 0;
  Index p = 0;
  bool sign = false;

  while (mult_ind[dim - 1] <= 1) {
    if (mult_ind[p] > 1) {
      scap -= 2 * w[p];
      mult_ind[p] = 0;
      ++p;
    } else {
      cw += (scap <= q) ? (sign ? -1 : 1) : 0;
      p = 0;
    }
    ++mult_ind[p];
    scap += w[p];
    sign ^= true;
  }
  return cw;
}

template <typename VectorT, typename MultiIndexSet>
void weightedTotalDegreeCombiDroplet(const VectorT &w, const Scalar q,
                                     MultiIndexSet &set) {
  const Index dim = w.size();
  typename MultiIndexSet::key_type mult_ind(dim);
  Scalar scap = 0;
  Index p = 0;
  Scalar sumw = 0;

  set.clear();
  std::fill(mult_ind.begin(), mult_ind.end(), 0);
  for (auto i = 0; i < dim; ++i) sumw += w[i];

  while (mult_ind[dim - 1] * w[dim - 1] <= q) {
    if (scap > q) {
      scap -= mult_ind[p] * w[p];
      mult_ind[p] = 0;
      ++p;
    } else {
      std::ptrdiff_t cw = 0;
      if (scap > q - sumw)
        cw = weightedTotalDegreecombiWeightDroplet(w, q - scap);
      if (cw) set.insert(std::make_pair(mult_ind, cw));
      p = 0;
    }
    mult_ind[p] += 1;
    scap += w[p];
  }

  return;
}

}  // namespace FMCA
#endif

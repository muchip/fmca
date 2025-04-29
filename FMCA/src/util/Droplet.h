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
 *  \brief enumerates all indices which are entriwise smaller or equal than
 *         a given multi index n;
 **/
template <typename MultiIndex, typename MultiIndexSet>
void TensorProductDroplet(const MultiIndex &n, MultiIndexSet &set) {
  const Index dim = n.size();
  MultiIndex index = n;
  std::fill(index.begin(), index.end(), 0);
  Index p = 0;
  set.clear();
  while (index[dim - 1] <= n[dim - 1]) {
    if (index[p] > n[p]) {
      index[p] = 0;
      ++p;
    } else {
      set.insert(index);
      p = 0;
    }
    index[p] += 1;
  }
  return;
}

}  // namespace FMCA
#endif

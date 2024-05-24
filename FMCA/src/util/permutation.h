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
#ifndef FMCA_UTIL_GETPERMUTATION_H_
#define FMCA_UTIL_GETPERMUTATION_H_

#include "Macros.h"

namespace FMCA {

template <typename Derived>
iVector permutationVector(const ClusterTreeBase<Derived> &ct) {
  return Eigen::Map<const FMCA::iVector>(ct.indices(), ct.block_size());
}

template <typename Derived>
iVector inversePermutationVector(const ClusterTreeBase<Derived> &ct) {
    iVector perm = permutationVector(ct);  // permutation Vector
    iVector inversePerm(perm.size());
    for(int i = 0; i < perm.size(); ++i) {
        inversePerm(perm(i)) = i;
    }    
    return inversePerm;
}


template <typename Derived>
Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, FMCA::Index>
permutationMatrix(const ClusterTreeBase<Derived> &ct) {
  return Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, FMCA::Index>(
      permutationVector(ct));
}
}  // namespace FMCA
#endif

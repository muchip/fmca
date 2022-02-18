// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_SAMPLETMAPPER_H_
#define FMCA_SAMPLETS_SAMPLETMAPPER_H_

namespace FMCA {

namespace internal {

template <typename Derived>
void sampletMapper(SampletTreeBase<Derived> &ST) {
  assert(ST.is_root() &&
         "sampletMapper needs to be called from the root cluster");
  IndexType sum = 0;
  // assign vector start_index to each wavelet cluster
  for (auto &&it : ST) {
    it.node().start_index_ = sum;
    sum += it.derived().nsamplets();
    if (it.is_root()) sum += it.derived().nscalfs();
  }
  assert(sum == ST.indices().size());
  return;
}

}  // namespace internal
}  // namespace FMCA
#endif

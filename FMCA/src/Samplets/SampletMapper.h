// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU General Public License version 3
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_SAMPLETMAPPER_H_
#define FMCA_SAMPLETS_SAMPLETMAPPER_H_

namespace FMCA {

namespace internal {

template <typename Derived>
void sampletMapper(SampletTreeBase<Derived> &ST) {
  assert(ST.is_root() &&
         "sampletMapper needs to be called from the root cluster");
  Index sum = 0;
  // assign vector start_index to each wavelet cluster
  for (auto &&it : ST) {
    it.node().start_index_ = sum;
    sum += it.derived().nsamplets();
    if (it.is_root()) sum += it.derived().nscalfs();
  }
  assert(sum == ST.indices().size());
  return;
}

template <typename Derived>
std::vector<Index> sampletLevelMapper(SampletTreeBase<Derived> &ST) {
  assert(ST.is_root() &&
         "sampletLevelMapper needs to be called from the root cluster");
  std::vector<Index> retval;
  // assign vector start_index to each wavelet cluster
  double geo_diam = ST.bb().col(2).norm();
  for (auto &&it : ST) {
    double diam = it.bb().col(2).norm();
    int level = std::round(-log(diam / geo_diam) / log(2));
    if (it.is_root())
      for (auto i = 0; i < it.derived().nscalfs(); ++i) retval.push_back(level);
    for (auto i = 0; i < it.derived().nsamplets(); ++i) retval.push_back(level);
  }
  return retval;
}

}  // namespace internal
}  // namespace FMCA
#endif

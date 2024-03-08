// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_SAMPLETADAPTIVECLUSTERING_H_
#define FMCA_SAMPLETS_SAMPLETADAPTIVECLUSTERING_H_

namespace FMCA {

template <typename Derived>
void markActiveClusters(std::vector<bool> &active,
                        const FMCA::SampletTreeBase<Derived> &st,
                        const FMCA::Vector &tdata, const FMCA::Scalar max_coeff,
                        const FMCA::Scalar threshold) {
  if (st.nSons()) {
    for (FMCA::Index i = 0; i < st.nSons(); ++i) {
      markActiveClusters(active, st.sons(i), tdata, max_coeff, threshold);
      active[st.block_id()] = active[st.sons(i).block_id()] ? true : false;
    }
  }
  const FMCA::Index ndist = st.is_root() ? st.Q().cols() : st.nsamplets();
  const FMCA::Scalar color =
      tdata.segment(st.start_index(), ndist).cwiseAbs().maxCoeff();
  if (color > threshold * max_coeff) active[st.block_id()] = true;

  return;
}

template <typename Derived>
void getActiveLeafs(std::vector<const Derived *> &active_leafs,
                    const std::vector<bool> &active,
                    const FMCA::SampletTreeBase<Derived> &st) {
  if (st.nSons()) {
    FMCA::Index scounter = 0;
    for (FMCA::Index i = 0; i < st.nSons(); ++i)
      if (active[st.sons(i).block_id()]) {
        ++scounter;
        getActiveLeafs(active_leafs, active, st.sons(i));
      }
    // the current cluster is an active leaf
    if (!scounter && active[st.block_id()])
      active_leafs.push_back(std::addressof(st.derived()));
    // there is at least one active child
    else if (scounter < st.nSons())
      for (FMCA::Index i = 0; i < st.nSons(); ++i)
        if (not active[st.sons(i).block_id()])
          active_leafs.push_back(std::addressof(st.sons(i).derived()));
  } else if (active[st.block_id()])
    active_leafs.push_back(std::addressof(st.derived()));

  return;
}
}  // namespace FMCA
#endif

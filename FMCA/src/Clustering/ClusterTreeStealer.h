// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_CLUSTERING_CLUSTERTREESTEALER_H_
#define FMCA_CLUSTERING_CLUSTERTREESTEALER_H_

namespace FMCA {
namespace internal {
/**
 *  \ingroup Clustering
 *  \brief moves the cluster structure of src node by node into dst. src keeps
 *  its skeleton only and should be discarded afterwards
 **/
struct ClusterTreeStealer {
  template <typename DstDerived, typename SrcDerived>
  static void steal(ClusterTreeBase<DstDerived> &dst,
                    ClusterTreeBase<SrcDerived> &src) {
    dst.node().bb_ = std::move(src.node().bb_);
    dst.node().c_ = std::move(src.node().c_);
    dst.node().r_ = src.node().r_;
    dst.node().indices_ = std::move(src.node().indices_);
    dst.node().indices_begin_ = src.node().indices_begin_;
    dst.node().block_id_ = src.node().block_id_;
    dst.node().block_size_ = src.node().block_size_;
    dst.appendSons(src.nSons());
    for (Index i = 0; i < src.nSons(); ++i) steal(dst.sons(i), src.sons(i));
    return;
  }
};
}  // namespace internal
}  // namespace FMCA
   //
#endif

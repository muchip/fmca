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
#ifndef FMCA_CLUSTERING_INITMETISCLUSTERTREEIMPL_H_
#define FMCA_CLUSTERING_INITMETISCLUSTERTREEIMPL_H_

namespace FMCA {
namespace internal {
/** \ingroup internal
 *  \brief initializes a bounding box for the geometry
 **/
template <>
struct ClusterTreeInitializer<MetisClusterTree> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename Graph>
  static void init(ClusterTreeBase<Derived> &CT, Index min_csize, Graph G) {
    CT.node().indices_begin_ = 0;
    CT.node().indices_ = std::shared_ptr<Index>(new Index[G.nnodes()],
                                                std::default_delete<Index[]>());
    CT.node().block_size_ = G.nnodes();
    Index *indices = CT.node().indices_.get();
    for (Index i = 0; i < CT.block_size(); ++i) indices[i] = G.labels()[i];
    init_ClusterTree_impl(CT, min_csize, G);
    Index i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  /** \ingroup internal
   *  \brief perform cluster refinement given a Splitter class
   **/
  template <typename Derived, typename Graph>
  static void init_ClusterTree_impl(ClusterTreeBase<Derived> &CT,
                                    Index min_csize, Graph &G) {
    typename traits<Derived>::Splitter split;
    const Index split_threshold = min_csize >= 1 ? (2 * min_csize - 1) : 1;
    if (CT.node().block_size_ > split_threshold) {
      CT.appendSons(2);
      CT.sons(0).node().indices_ = CT.node().indices_;
      CT.sons(1).node().indices_ = CT.node().indices_;
      std::vector<idx_t> part = partitionGraph(G);
      Graph G1 = G.split(part);
      CT.sons(0).node().indices_begin_ = CT.node().indices_begin_;
      CT.sons(0).node().block_size_ = G.labels().size();
      {
        Index *indices = CT.sons(0).node().indices_.get();
        for (Index i = 0; i < CT.sons(0).block_size(); ++i)
          indices[CT.sons(0).node().indices_begin_ + i] = G.labels()[i];
      }
      CT.sons(1).node().indices_begin_ =
          CT.node().indices_begin_ + CT.sons(0).block_size();
      CT.sons(1).node().block_size_ = G1.labels().size();
      {
        Index *indices = CT.sons(1).node().indices_.get();
        for (Index i = 0; i < CT.sons(1).block_size(); ++i)
          indices[CT.sons(1).node().indices_begin_ + i] = G1.labels()[i];
      }
      init_ClusterTree_impl<Derived>(CT.sons(0), min_csize, G);
      init_ClusterTree_impl<Derived>(CT.sons(1), min_csize, G1);
    }
    return;
  }
};
}  // namespace internal

}  // namespace FMCA

#endif

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
#ifndef FMCA_CLUSTERING_INITUNITKDTREEIMPL_H_
#define FMCA_CLUSTERING_INITUNITKDTREEIMPL_H_

namespace FMCA {
namespace internal {
/** \ingroup internal
 *  \brief initializes a bounding box for the geometry
 **/
template <>
struct ClusterTreeInitializer<UnitKDTree> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  static void init(ClusterTreeBase<Derived> &CT, Index min_csize,
                   const Matrix &P, Index n_levels) {
    CT.node().bb_.resize(P.rows(), 3);
    CT.node().bb_.col(0).setZero();
    CT.node().bb_.col(1).setOnes();
    CT.node().bb_.col(2).setOnes();
    CT.node().indices_begin_ = 0;
    CT.node().indices_ = std::shared_ptr<Index>(new Index[P.cols()],
                                                std::default_delete<Index[]>());
    CT.node().block_size_ = P.cols();
    Index *indices = CT.node().indices_.get();
    for (Index i = 0; i < CT.block_size(); ++i) indices[i] = i;
    init_ClusterTree_impl(CT, n_levels, P);
    Index i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  /** \ingroup internal
   *  \brief perform cluster refinement given a Splitter class
   *
   *  The leaf id is the row-major cell index and serves only as a key for
   *  O(1) point-to-leaf assignment. The index array is laid out in the order
   *  in which the DFS visits the leaves (sons in increasing order), so that
   *  every subtree owns one contiguous slice and the post-order pass
   *  (begin = min over sons, size = sum over sons) is exact for any d.
   **/
  template <typename Derived>
  static void init_ClusterTree_impl(ClusterTreeBase<Derived> &CT,
                                    Index n_levels, const Matrix &P) {
    const Index d = P.rows();
    const Index k = 1 << d;
    const Index n = 1 << n_levels;
    std::vector<Derived *> queue;
    std::vector<Derived *> dfs_order;
    std::vector<Derived *> leaf_nodes;  // leaves in DFS (son) order
    std::vector<Index> leaf_ids;        // their row-major cell ids
    if (n_levels > 0) queue.push_back(std::addressof(CT.derived()));
    while (queue.size()) {
      // get node from the queue and remember it for reverse dfs
      Derived &node = *(queue.back());
      dfs_order.push_back(std::addressof(node));
      queue.pop_back();
      // append children as they are needed
      node.appendSons(k);
      // set up bounding boxes for sons
      const Vector center = 0.5 * (node.bb().col(0) + node.bb().col(1));
      for (Index i = 0; i < k; ++i) {
        node.sons(i).node().bb_ = node.node().bb_;
        // set bounding boxes of children
        for (Index j = 0; j < d; ++j) {
          if ((i >> j) & 1) {
            node.sons(i).node().bb_(j, 0) = center(j);
            node.sons(i).node().bb_(j, 1) = node.bb()(j, 1);
          } else {
            node.sons(i).node().bb_(j, 0) = node.bb()(j, 0);
            node.sons(i).node().bb_(j, 1) = center(j);
          }
        }
        node.sons(i).node().bb_.col(2) =
            node.sons(i).node().bb_.col(1) - node.sons(i).node().bb_.col(0);
        // for now copy all references as they are for the parent
        node.sons(i).node().indices_ = node.node().indices_;
        node.sons(i).node().block_size_ = 0;
        node.sons(i).node().indices_begin_ = 0;
      }
      if (node.level() < n_levels - 1) {
        // push in reverse so that the stack pops the sons in order 0..k-1
        for (int i = k - 1; i >= 0; --i)
          queue.push_back(std::addressof(node.sons(i)));
      } else {
        for (Index i = 0; i < k; ++i) {
          Index coord = n * node.sons(i).node().bb_(0, 0);
          coord = std::max<Index>(0, std::min<Index>(n - 1, coord));
          Index id = coord;

          for (Index j = 1; j < d; ++j) {
            coord = Index(n * node.sons(i).node().bb_(j, 0));
            coord = std::max<Index>(0, std::min<Index>(n - 1, coord));
            id = n * id + coord;
          }
          leaf_nodes.push_back(std::addressof(node.sons(i)));
          leaf_ids.push_back(id);
        }
      }
    }
    // now perform point assignment to each leave and determine leave size
    std::vector<Index> leaf_id(P.cols());
    std::vector<Index> leaf_count(1 << (d * n_levels), 0);
    for (Index i = 0; i < P.cols(); ++i) {
      Index coord = n * P(0, i);
      coord = std::max<Index>(0, std::min<Index>(n - 1, coord));
      Index id = coord;
      for (Index j = 1; j < P.rows(); ++j) {
        coord = n * P(j, i);
        coord = std::max<Index>(0, std::min<Index>(n - 1, coord));
        id = n * id + coord;
      }
      leaf_id[i] = id;
      ++(leaf_count[id]);
    }
    // hand out slices in DFS (son) order, remember the offset per cell id
    std::vector<Index> leaf_offset(leaf_count.size(), 0);
    Index offset = 0;
    for (Index l = 0; l < leaf_nodes.size(); ++l) {
      const Index id = leaf_ids[l];
      leaf_nodes[l]->node().block_size_ = leaf_count[id];
      leaf_nodes[l]->node().indices_begin_ = offset;
      leaf_offset[id] = offset;
      offset += leaf_count[id];
    }
    // copy everything in place
    for (Index i = 0; i < leaf_id.size(); ++i)
      CT.node().indices_.get()[(leaf_offset[leaf_id[i]])++] = i;
    // finally traverse tree in post order to set parents
    for (auto it = dfs_order.rbegin(); it != dfs_order.rend(); ++it) {
      Derived &node = *(*it);
      Index begin = P.cols();
      Index size = 0;
      for (Index i = 0; i < node.nSons(); ++i) {
        begin = begin > node.sons(i).indices_begin()
                    ? node.sons(i).indices_begin()
                    : begin;
        size += node.sons(i).block_size();
      }
      node.node().indices_begin_ = begin;
      node.node().block_size_ = size;
    }
    return;
  }
};
}  // namespace internal

}  // namespace FMCA

#endif

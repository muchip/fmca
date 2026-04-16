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
///
#ifndef FMCA_CLUSTERING_UNITKDTREE_H_
#define FMCA_CLUSTERING_UNITKDTREE_H_

namespace FMCA {

struct UnitKDTreeNode : public ClusterTreeNodeBase<UnitKDTreeNode> {};

namespace internal {
template <>
struct traits<UnitKDTree> {
  typedef ClusterTreeNode Node;
  typedef ClusterSplitter::CardinalityBisection Splitter;
};
}  // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct UnitKDTree : public ClusterTreeBase<UnitKDTree> {
  typedef ClusterTreeBase<UnitKDTree> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  using initializer = internal::ClusterTreeInitializer<UnitKDTree>;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  UnitKDTree() {}
  UnitKDTree(const Matrix &P, Index n_levels = 1) { init(P, n_levels); }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix &P, Index n_levels = 1) {
    initializer::init(*this, 0, P, n_levels);
  }
};

}  // namespace FMCA
#endif

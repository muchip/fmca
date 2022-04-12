// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_CLUSTERING_CLUSTERTREE_H_
#define FMCA_CLUSTERING_CLUSTERTREE_H_

namespace FMCA {

namespace internal {
template <> struct traits<ClusterTreeNode> {
  typedef FloatType value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
};
} // namespace internal

struct ClusterTreeNode : public ClusterTreeNodeBase<ClusterTreeNode> {};

namespace internal {
template <> struct traits<ClusterTree> {
  typedef FloatType value_type;
  typedef ClusterTreeNode node_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef ClusterSplitter::CardinalityBisection<value_type> Splitter;
};
} // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct ClusterTree : public ClusterTreeBase<ClusterTree> {
  typedef typename internal::traits<ClusterTree>::eigenMatrix eigenMatrix;
  typedef ClusterTreeBase<ClusterTree> Base;
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
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTree() {}
  ClusterTree(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    init(P, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  void init(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    internal::ClusterTreeInitializer<ClusterTree>::init(*this, min_cluster_size,
                                                        P);
  }
};

} // namespace FMCA
#endif

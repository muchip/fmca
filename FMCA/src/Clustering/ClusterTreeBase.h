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
#ifndef FMCA_CLUSTERING_CLUSTERTREEBASE_H_
#define FMCA_CLUSTERING_CLUSTERTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief ClusterTreeNodeBase defines the basic fields required for an
 *         abstract ClusterTree, i.e. bounding box and indices and global
 *         index position
 **/
template <typename Derived>
struct ClusterTreeNodeBase : public NodeBase<Derived> {
  ClusterTreeNodeBase() : indices_begin_(-1) {
    bb_.resize(0, 0);
    indices_.resize(0);
  }
  typename internal::traits<Derived>::eigenMatrix bb_;
  std::vector<IndexType> indices_;
  IndexType indices_begin_;
};

/**
 *  \ingroup Clustering
 *  \brief The ClusterTreeBase class manages abstract cluster trees
 *         that may be described by subdivision of index sets and bounding
 *         boxes
 **/
template <typename Derived>
struct ClusterTreeBase : public TreeBase<Derived> {
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef TreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;

  void init(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    internal::init_BoundingBox_impl(*this, P, min_cluster_size);
    node().indices_begin_ = 0;
    node().indices_.resize(P.cols());
    std::iota(node().indices_.begin(), node().indices_.end(), 0u);
    internal::init_ClusterTree_impl(*this, P, min_cluster_size);
    internal::shrinkToFit_impl(*this, P);
  }
  //////////////////////////////////////////////////////////////////////////////
  // getter
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &bb() const { return node().bb_; }

  const std::vector<IndexType> &indices() const { return node().indices_; }

  IndexType indices_begin() const { return node().indices_begin_; }
};

}  // namespace FMCA
#endif

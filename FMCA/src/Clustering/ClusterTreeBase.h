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
 *         abstract ClusterTree, i.e. bounding box and indices, global
 *         index position and block id
 **/
template <typename Derived>
struct ClusterTreeNodeBase : public NodeBase<Derived> {
  ClusterTreeNodeBase() : indices_begin_(-1), block_id_(-1) {
    bb_.resize(0, 0);
    indices_.resize(0);
  }
  Matrix bb_;
  std::vector<Index> indices_;
  Index indices_begin_;
  Index block_id_;
};

/**
 *  \ingroup Clustering
 *  \brief The ClusterTreeBase class manages abstract cluster trees
 *         that may be described by subdivision of index sets and bounding
 *         boxes
 **/
template <typename Derived> struct ClusterTreeBase : public TreeBase<Derived> {
  typedef typename internal::traits<Derived>::Node Node;
  typedef TreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::derived;
  using Base::init;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  //////////////////////////////////////////////////////////////////////////////
  // getter
  //////////////////////////////////////////////////////////////////////////////
  const Matrix &bb() const { return node().bb_; }

  const std::vector<Index> &indices() const { return node().indices_; }

  Index indices_begin() const { return node().indices_begin_; }

  Index block_id() const { return node().block_id_; }
};

} // namespace FMCA
#endif

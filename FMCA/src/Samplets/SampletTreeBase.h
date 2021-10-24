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
#ifndef FMCA_SAMPLETS_SAMPLETTREEBASE_H_
#define FMCA_SAMPLETS_SAMPLETTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief ClusterTreeNodeBase defines the basic fields required for an
 *         abstract ClusterTree, i.e. bounding box and indices and global
 *         index position
 **/
template <typename Derived>
struct SampletTreeNodeBase : public ClusterTreeNodeBase<Derived> {
  ClusterTreeNodeBase() : indices_begin_(-1) {
    bb_.resize(0, 0);
    indices_.resize(0);
  }
  typename internal::traits<Derived>::eigenMatrix Q_;
  typename internal::traits<Derived>::eigenMatrix mom_buffer_;
  IndexType nscalfs_;
  IndexType nsamplets_;
  IndexType samplet_level_;
  IndexType start_index_;
  IndexType block_id_;
};

/**
 *  \ingroup Clustering
 *  \brief The ClusterTreeBase class manages abstract cluster trees
 *         that may be described by subdivision of index sets and bounding
 *         boxes
 **/
template <typename Derived>
struct SampletTreeBase : public ClusterTreeBase<Derived> {
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef TreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;

  void init(const eigenMatrix &P, IndexType dtilde = 1,
            IndexType min_cluster_size = 1) {
    Base::init(P, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
};

} // namespace FMCA
#endif

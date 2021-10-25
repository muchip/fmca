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
 *  \ingroup Samplets
 *  \brief SampletTreeNodeBase defines the basic fields required for an
 *         abstract SampletTree, i.e. the transformation matrices
 **/
template <typename Derived>
struct SampletTreeNodeBase : public ClusterTreeNodeBase<Derived> {
  typename internal::traits<Derived>::eigenMatrix Q_;
  typename internal::traits<Derived>::eigenMatrix mom_buffer_;
  IndexType nscalfs_;
  IndexType nsamplets_;
  IndexType start_index_;
  IndexType block_id_;
};

/**
 *  \ingroup Samplets
 *  \brief The SampletTreeBase class manages abstract samplet trees
 **/
template <typename Derived>
struct SampletTreeBase : public ClusterTreeBase<Derived> {
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef ClusterTreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::init;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;

  //////////////////////////////////////////////////////////////////////////////
  IndexType nscalfs() const { return node().nscalfs_; }
  IndexType nsamplets() const { return node().nsamplets_; }
  IndexType block_id() const { return node().block_id_; }
  IndexType start_index() const { return node().start_index_; }
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &Q() const { return node().Q_; }
};

}  // namespace FMCA
#endif

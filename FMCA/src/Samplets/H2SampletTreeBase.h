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
#ifndef FMCA_SAMPLETS_H2SAMPLETTREEBASE_H_
#define FMCA_SAMPLETS_H2SAMPLETTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief H2SampletTreeNodeBase defines the basic fields required for an
 *         abstract H2SampletTree by combining the fields for ClusterTree,
 *         H2ClusterTree and SampletTree
 **/

template <typename Derived>
struct H2SampletTreeNodeBase : public ClusterTreeNodeBase<Derived>,
                             public H2ClusterTreeNodeDataFields<Derived>,
                             public SampletTreeNodeDataFields<Derived> {};
/**
 *  \ingroup Samplets
 *  \brief The SampletTreeBase class manages abstract samplet trees
 **/
template <typename Derived>
struct H2SampletTreeBase : public H2ClusterTreeBase<Derived>,
                           public SampletTreeBase<Derived> {
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef H2ClusterTreeBase<Derived> Base;
  // make base class methods explicitly visible
  using TreeBase<Derived>::is_root;
  using Base::appendSons;
  using Base::indices;
  using Base::indices_begin;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
};
} // namespace FMCA
#endif

// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_WEDGELETS_WEDGELETTREEBASE_H_
#define FMCA_WEDGELETS_WEDGELETTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Wedgelets
 *  \brief WedgeletTreeNodeBase defines the basic fields required for a
 *         WedgeletTree, i.e. fields for a ClusterTree
 *         Landmark, Interpolation points, dimension, polynomial degree,
 *         coefficients
 **/
template <typename Derived>
struct WedgeletTreeNodeDataFields {
  Index landmark_;
  Index dim_;
  Index deg_;
  Matrix C_;
  Scalar err_;
};

template <typename Derived>
struct WedgeletTreeNodeBase : public ClusterTreeNodeBase<Derived>,
                              public WedgeletTreeNodeDataFields<Derived> {};
/**
 *  \ingroup H2Matrix
 *  \brief The WedgeletTreeBase class manages an abstract H2-matrix
 *         cluster tree
 */

template <typename Derived>
struct WedgeletTreeBase : public ClusterTreeBase<Derived> {
  typedef typename internal::traits<Derived>::Node Node;
  typedef ClusterTreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::init;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
};

}  // namespace FMCA
#endif

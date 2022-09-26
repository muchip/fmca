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
#ifndef FMCA_H2MATRIX_H2CLUSTERTREEBASE_H_
#define FMCA_H2MATRIX_H2CLUSTERTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief H2ClusterTreeNodeBase defines the basic fields required for an
 *         H2ClusterTree, i.e. fields for a ClusterTree plus transfer matrices
 *         and cluster bases
 **/
template <typename Derived>
struct H2ClusterTreeNodeDataFields {
  std::vector<Matrix> E_;
  Matrix V_;
};

template <typename Derived>
struct H2ClusterTreeNodeBase : public ClusterTreeNodeBase<Derived>,
                               public H2ClusterTreeNodeDataFields<Derived> {};
/**
 *  \ingroup H2Matrix
 *  \brief The H2ClusterTreeBase class manages an abstract H2-matrix
 *         cluster tree
 */

template <typename Derived>
struct H2ClusterTreeBase : public ClusterTreeBase<Derived> {
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

  const Matrix &V() const { return node().V_; }
  Matrix &V() { return node().V_; }

  const std::vector<Matrix> &Es() const { return node().E_; }
  std::vector<Matrix> &Es() { return node().E_; }
};

}  // namespace FMCA
#endif

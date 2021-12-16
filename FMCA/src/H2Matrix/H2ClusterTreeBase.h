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
  std::vector<typename internal::traits<Derived>::eigenMatrix> E_;
  typename internal::traits<Derived>::eigenMatrix V_;
  std::shared_ptr<typename internal::traits<Derived>::Interpolator> interp_;
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
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef typename internal::traits<Derived>::Interpolator Interpolator;
  typedef ClusterTreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;

  void init(const eigenMatrix &P, IndexType min_cluster_size = 1,
            IndexType polynomial_degree = 3) {
    // init interpolation routines
    node().interp_ = std::make_shared<Interpolator>();
    node().interp_->init(P.rows(), polynomial_degree);
    // init cluster tree first
    Base::init(P, min_cluster_size > node().interp_->Xi().cols()
                      ? min_cluster_size
                      : node().interp_->Xi().cols());

    internal::compute_cluster_bases_impl<Interpolator, Derived, eigenMatrix>(
        *this, P);
  }

  const eigenMatrix &V() const { return node().V_; }
  eigenMatrix &V() { return node().V_; }

  const std::vector<eigenMatrix> &Es() const { return node().E_; }
  std::vector<eigenMatrix> &Es() { return node().E_; }

  const eigenMatrix &Xi() const { return node().interp_->Xi(); }
  //////////////////////////////////////////////////////////////////////////////
};

}  // namespace FMCA
#endif

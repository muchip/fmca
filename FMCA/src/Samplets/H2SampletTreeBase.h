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
 *  \brief The H2SampletTreeBase class manages abstract samplet trees
 *         We replicate the H2ClusterTree init here to avoid a diamond
 **/
template <typename Derived>
struct H2SampletTreeBase : public SampletTreeBase<Derived> {
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::Interpolator Interpolator;
  typedef SampletTreeBase<Derived> Base;
  // make base class methods explicitly visible
  using TreeBase<Derived>::is_root;
  using Base::appendSons;
  using Base::indices;
  using Base::indices_begin;
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
  void computeMultiscaleClusterBasis() {
    if (!nSons()) {
      node().V_ = node().V_ * node().Q_;
    } else {
      // compute multiscale cluster bases of sons and update own
      for (auto i = 0; i < nSons(); ++i)
        sons(i).computeMultiscaleClusterBasis();
      node().V_.resize(0, 0);
      for (auto i = 0; i < nSons(); ++i) {
        node().V_.conservativeResize(sons(i).V().rows(),
                                     node().V_.cols() + sons(i).nscalfs());
        node().V_.rightCols(sons(i).nscalfs()) =
            node().E_[i] * sons(i).V().leftCols(sons(i).nscalfs());
      }
      node().V_ *= node().Q_;
    }
    return;
  }
  void updateMultiscaleClusterBasis() {
    // compute multiscale cluster bases of sons and update own
    for (auto i = 0; i < nSons(); ++i) sons(i).updateMultiscaleClusterBasis();
    node().V_ = node().interp_->invV().transpose() * node().V_;
    return;
  }
};
}  // namespace FMCA
#endif

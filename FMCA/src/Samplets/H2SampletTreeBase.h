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

#include "../H2Matrix/H2ClusterTreeBase.h"

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
  typedef typename internal::traits<Derived>::Node Node;
  typedef SampletTreeBase<Derived> Base;
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
  using Base::nsamplets;
  using Base::nscalfs;
  using Base::nSons;
  using Base::Q;
  using Base::sons;
  using Base::start_index;

  const Matrix &V() const { return node().V_; }
  Matrix &V() { return node().V_; }

  const std::vector<Matrix> &Es() const { return node().E_; }
  std::vector<Matrix> &Es() { return node().E_; }

  const Matrix &Xi() const { return node().interp_->Xi(); }
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
  template <typename Moments>
  void updateMultiscaleClusterBasis(const Moments &mom) {
    // compute multiscale cluster bases of sons and update own
    for (auto i = 0; i < nSons(); ++i)
      sons(i).updateMultiscaleClusterBasis(mom);
    node().V_ = mom.interp().invV().transpose() * node().V_;
    return;
  }
};
} // namespace FMCA
#endif

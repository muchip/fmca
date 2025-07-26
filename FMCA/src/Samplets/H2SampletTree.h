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

#ifndef FMCA_SAMPLETS_H2SAMPLETTREE_H_
#define FMCA_SAMPLETS_H2SAMPLETTREE_H_

#include "../H2Matrix/compute_cluster_bases_impl.h"

namespace FMCA {

struct H2SampletTreeNode : public H2SampletTreeNodeBase<H2SampletTreeNode> {};

namespace internal {
template <typename ClusterTreeType>
struct traits<H2SampletTree<ClusterTreeType>>
    : public traits<SampletTree<ClusterTreeType>> {
  typedef H2SampletTreeNode Node;
};
}  // namespace internal

/**
 *  \ingroup Samplets
 *  \brief The SampletTree class manages samplets constructed on a cluster tree.
 */
template <typename ClusterTreeType>
class H2SampletTree : public H2SampletTreeBase<H2SampletTree<ClusterTreeType>> {
 public:
  typedef ClusterTreeType ClusterTree;
  typedef typename internal::traits<H2SampletTree>::Node Node;
  typedef H2SampletTreeBase<H2SampletTree<ClusterTreeType>> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::Es;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nsamplets;
  using Base::nscalfs;
  using Base::nSons;
  using Base::Q;
  using Base::sons;
  using Base::start_index;
  using Base::V;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2SampletTree() {}
  template <typename Moments, typename SampletMoments, typename... Ts>
  H2SampletTree(const Moments &mom, const SampletMoments &smom,
                Index min_cluster_size, Ts &&...ts) {
    init(mom, smom, min_cluster_size, std::forward<Ts>(ts)...);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename Moments, typename SampletMoments, typename... Ts>
  void init(const Moments &mom, const SampletMoments &smom,
            Index min_cluster_size, Ts &&...ts) {
    eigen_assert(mom.interp().Xi().cols() >= smom.mdtilde() &&
                 "There have to be as many multipole moms as samplet moms");
    const Index mincsize = min_cluster_size > mom.interp().Xi().cols()
                               ? min_cluster_size
                               : mom.interp().Xi().cols();
    // init cluster tree
    ClusterTreeType::initializer::init(*this, mincsize,
                                       std::forward<Ts>(ts)...);
    // init hierarchical cluster basis
    internal::compute_cluster_bases_impl::compute(*this, mom);
    // internal::compute_cluster_bases_impl::check_transfer_matrices(*this,
    // mom);
    // init samplet basis
    computeSamplets(smom);
    internal::sampletMapper<H2SampletTree>(*this);

    Base::computeMultiscaleClusterBasis();

    return;
  }

 private:
  template <typename Moments>
  void computeSamplets(const Moments &mom) {
    if (nSons()) {
      Index offset = 0;
      for (auto i = 0; i < nSons(); ++i) {
        sons(i).computeSamplets(mom);
        // the son now has moments, lets grep them...
        Matrix shift = 0.5 * (sons(i).bb().col(0) - bb().col(0) +
                              sons(i).bb().col(1) - bb().col(1));
        node().mom_buffer_.conservativeResize(
            sons(i).node().mom_buffer_.rows(),
            offset + sons(i).node().mom_buffer_.cols());
        node().mom_buffer_.block(0, offset, sons(i).node().mom_buffer_.rows(),
                                 sons(i).node().mom_buffer_.cols()) =
            mom.shift_matrix(shift) * sons(i).node().mom_buffer_;
        offset += sons(i).node().mom_buffer_.cols();
        // clear moment buffer of the children
        sons(i).node().mom_buffer_.resize(0, 0);
      }
    } else
      // compute cluster basis of the leaf
      node().mom_buffer_ = mom.moment_matrix(*this);
    // are there samplets?
    if (mom.mdtilde() < node().mom_buffer_.cols()) {
      Eigen::HouseholderQR<Matrix> qr(node().mom_buffer_.transpose());
      node().Q_ = qr.householderQ();
      node().nscalfs_ = mom.mdtilde();
      node().nsamplets_ = node().Q_.cols() - node().nscalfs_;
      // this is the moment for the dad cluster
      node().mom_buffer_ = qr.matrixQR()
                               .block(0, 0, mom.mdtilde(), mom.mdtilde2())
                               .template triangularView<Eigen::Upper>()
                               .transpose();
    } else {
      node().Q_ = Matrix::Identity(node().mom_buffer_.cols(),
                                   node().mom_buffer_.cols());
      node().nscalfs_ = node().mom_buffer_.cols();
      node().nsamplets_ = 0;
    }
    return;
  }
};
}  // namespace FMCA
#endif

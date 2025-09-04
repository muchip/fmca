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
///
#ifndef FMCA_CLUSTERING_SPHERECLUSTERTREE_H_
#define FMCA_CLUSTERING_SPHERECLUSTERTREE_H_

namespace FMCA {

struct SphereClusterTreeNode
    : public ClusterTreeNodeBase<SphereClusterTreeNode> {};

namespace internal {
template <>
struct traits<SphereClusterTree> {
  typedef SphereClusterTreeNode Node;
  typedef ClusterSplitter::GeometricKDSplitting Splitter;
};
}  // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct SphereClusterTree : public ClusterTreeBase<SphereClusterTree> {
  typedef ClusterTreeBase<SphereClusterTree> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  using initializer = internal::ClusterTreeInitializer<SphereClusterTree>;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  SphereClusterTree() {}
  SphereClusterTree(const Matrix &P, Index min_csize = 1) {
    init(P, min_csize);
  }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix &P, Index min_csize = 1) {
    // we initialize the tree as a classical 3D cluster tree. Afterwards, we
    // assign centers and radii
    initializer::init(*this, min_csize, P);
  }

  static Scalar geodesicDistance(const Vector &a, const Vector &b) {
    const Scalar dot = a.dot(b);
    const Scalar clamped_dot = std::min(1., std::max(-1., dot));
    return std::acos(clamped_dot);
  }

  const Vector &center() const { return this->node().c_; }
  const Scalar radius() const { return this->node().r_; }

 private:
};

}  // namespace FMCA
#endif

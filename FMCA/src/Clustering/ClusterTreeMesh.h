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
#ifndef FMCA_CLUSTERING_CLUSTERTREEMESH_H_
#define FMCA_CLUSTERING_CLUSTERTREEMESH_H_

namespace FMCA {

namespace internal {
template <> struct traits<ClusterTreeMesh> {
  typedef ClusterTreeNode Node;
  typedef ClusterSplitter::GeometricBisection Splitter;
};
} // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct ClusterTreeMesh : public ClusterTreeBase<ClusterTreeMesh> {
  typedef ClusterTreeBase<ClusterTreeMesh> Base;
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
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTreeMesh() {}
  ClusterTreeMesh(const Matrix &V, const iMatrix &F,
                  Index min_cluster_size = 1) {
    init(V, F, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix &V, const iMatrix &F, Index min_cluster_size = 1) {

    internal::ClusterTreeInitializer<ClusterTreeMesh>::init(
        *this, min_cluster_size, V, F);
  }
};

} // namespace FMCA
#endif

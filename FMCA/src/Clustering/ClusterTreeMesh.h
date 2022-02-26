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
  typedef FloatType value_type;
  typedef ClusterTreeNode node_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef ClusterSplitter::CardinalityBisection<value_type> Splitter;
};
} // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct ClusterTreeMesh : public ClusterTreeBase<ClusterTreeMesh> {
  typedef typename internal::traits<ClusterTree>::eigenMatrix eigenMatrix;
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
  template <typename Derived, typename otherDerived>
  ClusterTreeMesh(const Eigen::MatrixBase<Derived> &V,
                  const Eigen::MatrixBase<otherDerived> &F,
                  IndexType min_cluster_size = 1) {
    init(V, F, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename otherDerived>
  void init(const Eigen::MatrixBase<Derived> &V,
            const Eigen::MatrixBase<otherDerived> &F,
            IndexType min_cluster_size = 1) {
    // generate list of element centers of gravity
    eigenMatrix P(V.cols(), F.rows());
    P.setZero();
    for (auto i = 0; i < P.cols(); ++i)
      for (auto j = 0; j < F.cols(); ++j)
        P.col(i) += V.row(F(i, j)).transpose() / F.cols();
    internal::ClusterTreeInitializer<ClusterTree>::init(*this, P,
                                                        min_cluster_size);
  }
};

} // namespace FMCA
#endif

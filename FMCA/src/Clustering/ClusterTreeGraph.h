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
#ifndef FMCA_CLUSTERING_CLUSTERTREEGRAPH_H_
#define FMCA_CLUSTERING_CLUSTERTREEGRAPH_H_

namespace FMCA {

namespace internal {
template <>
struct traits<ClusterTreeGraph> {
  typedef FloatType value_type;
  typedef IndexType idx_type;
  typedef ClusterTreeNode node_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef Eigen::Matrix<idx_type, Eigen::Dynamic, Eigen::Dynamic> idxMatrix;
  typedef ClusterSplitter::MetisBisection<eigenMatrix, idxMatrix,
                                          Eigen::MatrixXi, const bool>
      Splitter;
};
}  // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct ClusterTreeGraph : public ClusterTreeBase<ClusterTreeGraph> {
  typedef ClusterTreeBase<ClusterTreeGraph> Base;
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
  ClusterTreeGraph() {}
  template <typename Derived, typename otherDerived, typename logicType>
  ClusterTreeGraph(const Eigen::MatrixBase<Derived> &V,
                   const Eigen::MatrixBase<otherDerived> &F,
                   IndexType min_cluster_size = 1,
                   const logicType dual = true) {
    init(V, F, min_cluster_size, dual);
  }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename otherDerived, typename logicType>
  void init(const Eigen::MatrixBase<Derived> &V,
            const Eigen::MatrixBase<otherDerived> &F,
            IndexType min_cluster_size = 1, const logicType dual = true) {
    internal::ClusterTreeInitializer<ClusterTreeGraph>::init(
        *this, min_cluster_size, V, F, dual);
  }
};

}  // namespace FMCA
#endif

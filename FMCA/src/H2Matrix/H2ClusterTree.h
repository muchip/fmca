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
#ifndef FMCA_H2MATRIX_H2CLUSTERTREE_H_
#define FMCA_H2MATRIX_H2CLUSTERTREE_H_

namespace FMCA {

namespace internal {
template <>
struct traits<H2ClusterTreeNode> {
  typedef FloatType value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef TotalDegreeInterpolator<value_type> Interpolator;
};
}  // namespace internal

struct H2ClusterTreeNode : public H2ClusterTreeNodeBase<H2ClusterTreeNode> {};

namespace internal {
template <>
struct traits<H2ClusterTree> : public traits<ClusterTree> {
  typedef H2ClusterTreeNode node_type;
  typedef traits<H2ClusterTreeNode>::Interpolator Interpolator;
};
}  // namespace internal

/**
 *  \ingroup H2Matrix
 *  \brief The H2ClusterTree class manages the cluster bases for a given
 *         ClusterTree.
 *
 *         The tree structure from the ClusterTree is replicated here. This
 *         was a design decision as a cluster tree per se is not related to
 *         cluster bases. Also note that we just use pointers to clusters here.
 *         Thus, if the cluster tree is mutated or goes out of scope, we get
 *         dangeling pointers!
 */
struct H2ClusterTree : public H2ClusterTreeBase<H2ClusterTree> {
 public:
  typedef typename internal::traits<H2ClusterTree>::value_type value_type;
  typedef typename internal::traits<H2ClusterTree>::node_type node_type;
  typedef typename internal::traits<H2ClusterTree>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<H2ClusterTree>::Interpolator Interpolator;
  typedef H2ClusterTreeBase<H2ClusterTree> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::Es;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  using Base::V;
  using Base::Xi;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2ClusterTree() {}
  H2ClusterTree(const eigenMatrix &P, IndexType min_cluster_size = 1,
                IndexType polynomial_degree = 3) {
    init(P, min_cluster_size, polynomial_degree);
  }
  //////////////////////////////////////////////////////////////////////////////
  void init(const eigenMatrix &P, IndexType min_cluster_size = 1,
            IndexType polynomial_degree = 3) {
    // init interpolation routines
    node().interp_ = std::make_shared<Interpolator>();
    node().interp_->init(P.rows(), polynomial_degree);
    // init cluster tree first
    const IndexType mincsize = min_cluster_size > node().interp_->Xi().cols()
                                   ? min_cluster_size
                                   : node().interp_->Xi().cols();
    internal::ClusterTreeInitializer<ClusterTree>::init(*this, P, mincsize);

    internal::compute_cluster_bases_impl<Interpolator, H2ClusterTree, eigenMatrix>(
        *this, P);
  }
};

}  // namespace FMCA
#endif

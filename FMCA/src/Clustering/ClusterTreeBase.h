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
#ifndef FMCA_CLUSTERING_CLUSTERTREEBASE_H_
#define FMCA_CLUSTERING_CLUSTERTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief ClusterTreeNodeBase defines the basic fields required for an
 *         abstract ClusterTree, i.e. bounding box and indices, global
 *         index position and block id
 **/
template <typename Derived>
struct ClusterTreeNodeBase : public NodeBase<Derived> {
  ClusterTreeNodeBase()
      : indices_(nullptr), indices_begin_(-1), block_id_(-1), block_size_(0) {
    bb_.resize(0, 0);
  }
  Matrix bb_;
  Vector c_;
  Scalar r_;
  std::shared_ptr<Index> indices_;
  Index indices_begin_;
  Index block_id_;
  Index block_size_;
};

/**
 *  \ingroup Clustering
 *  \brief The ClusterTreeBase class manages abstract cluster trees
 *         that may be described by subdivision of index sets and bounding
 *         boxes
 **/
template <typename Derived>
struct ClusterTreeBase : public TreeBase<Derived> {
  typedef typename internal::traits<Derived>::Node Node;
  typedef TreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::dad;
  using Base::derived;
  using Base::init;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  //////////////////////////////////////////////////////////////////////////////
  // getter
  //////////////////////////////////////////////////////////////////////////////
  const Matrix& bb() const { return node().bb_; }

  const Index* indices() const {
    return node().indices_.get() + node().indices_begin_;
  }

  Index indices_begin() const { return node().indices_begin_; }

  Index block_id() const { return node().block_id_; }

  Index block_size() const { return node().block_size_; }

  Matrix toClusterOrder(const Matrix& mat) const {
    Matrix retval = mat;
    for (FMCA::Index j = 0; j < block_size(); ++j)
      retval.row(j) = mat.row(indices()[j]);
    return retval;
  }

  Matrix toNaturalOrder(const Matrix& mat) const {
    Matrix retval = mat;
    for (FMCA::Index j = 0; j < block_size(); ++j)
      retval.row(indices()[j]) = mat.row(j);
    return retval;
  }
};

}  // namespace FMCA
#endif

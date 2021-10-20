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
#ifndef FMCA_CLUSTERTREE_CLUSTERTREE_H_
#define FMCA_CLUSTERTREE_CLUSTERTREE_H_

namespace FMCA {

template <typename ValueType>
struct ClusterTreeNode : public NodeBase<ClusterTreeNode<ValueType>> {
  ClusterTreeNode() : indices_begin_(-1) {
    bb_.resize(0, 0);
    indices_.resize(0);
  }
  Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> bb_;
  std::vector<IndexType> indices_;
  IndexType indices_begin_;
};

template <typename ValueType, typename Splitter> class ClusterTree;

namespace internal {

template <typename ValueType, typename TheSplitter>
struct traits<ClusterTree<ValueType, TheSplitter>> {
  typedef ValueType value_type;
  typedef ClusterTreeNode<value_type> node_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef TheSplitter Splitter;
};
} // namespace internal

/**
 *  \ingroup ClusterTree
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
template <typename ValueType,
          typename Splitter = ClusterSplitter::CardinalityBisection<ValueType>>
class ClusterTree : public ClusterTreeBase<ClusterTree<ValueType, Splitter>> {
  using ClusterTreeBase<ClusterTree<ValueType, Splitter>>::ClusterTreeBase;
};

using ClusterT = ClusterTree<double>;
} // namespace FMCA
#endif

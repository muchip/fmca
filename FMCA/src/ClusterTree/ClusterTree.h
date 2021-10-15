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
struct ClusterTreeData {
  ValueType geometry_diam_ = 0;
  IndexType max_id_ = 0;
  IndexType max_level_ = 0;
};

template <typename ValueType>
struct ClusterTreeNode : public NodeBase<ClusterTreeNode<ValueType>> {
  ClusterTreeNode() : tree_data_(nullptr), indices_begin_(-1) {
    bb_.resize(0, 0);
    indices_.resize(0);
  }
  std::shared_ptr<ClusterTreeData<ValueType>> tree_data_;
  Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> bb_;
  std::vector<IndexType> indices_;
  IndexType indices_begin_;
};

template <typename ValueType, typename Splitter>
class ClusterTree;

namespace internal {

template <typename ValueType, typename Splitter>
struct traits<ClusterTree<ValueType, Splitter>> {
  typedef ValueType value_type;
  typedef ClusterTreeNode<value_type> node_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
};
}  // namespace internal

/**
 *  \ingroup ClusterTree
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
template <typename ValueType,
          typename Splitter = ClusterSplitter::GeometricBisection<ValueType>>
class ClusterTree : public TreeBase<ClusterTree<ValueType>> {
 public:
  typedef typename internal::traits<ClusterTree>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<ClusterTree>::node_type node_type;
  // make base class methods visible
  using TreeBase<ClusterTree<ValueType>>::node;
  using TreeBase<ClusterTree<ValueType>>::sons;
  using TreeBase<ClusterTree<ValueType>>::appendSons;
  using TreeBase<ClusterTree<ValueType>>::nSons;
  using TreeBase<ClusterTree<ValueType>>::level;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTree() {}
  ClusterTree(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    init(P, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    // set up bounding box for root node
    node().tree_data_ = std::make_shared<ClusterTreeData<ValueType>>();
    initBoundingBox(P);
    node().indices_begin_ = 0;
    node().indices_.resize(P.cols());
    std::iota(node().indices_.begin(), node().indices_.end(), 0u);
    computeClusters(P, min_cluster_size);
    shrinkToFit(P);
    node().tree_data_->geometry_diam_ = node().bb_.col(2).norm();
  }
  //////////////////////////////////////////////////////////////////////////////
  // get a vector with the bounding boxes on a certain level
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVector(std::vector<eigenMatrix> *bbvec, IndexType lvl = 0) {
    if (nSons() && level() < lvl)
      for (auto i = 0; i < nSons(); ++i) sons(i).get_BboxVector(bbvec, lvl);
    if (level() == lvl)
      if (node().indices_.size()) bbvec->push_back(node().bb_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVectorLeafs(std::vector<eigenMatrix> *bbvec) {
    if (nSons())
      for (auto i = 0; i < nSons(); ++i) sons(i).get_BboxVectorLeafs(bbvec);
    else if (node().indices_.size())
      bbvec->push_back(node().bb_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // getter
  //////////////////////////////////////////////////////////////////////////////
  const ClusterTree *get_cluster() const { return this; }

  const eigenMatrix &get_bb() const { return node().bb_; }

  const std::vector<IndexType> &get_indices() const { return node().indices_; }

  const ClusterTreeData<ValueType> &get_tree_data() const {
    return *(node().tree_data_);
  }

  IndexType get_level() const { return level(); }

  IndexType get_indices_begin() const { return node().indices_begin_; }
  //////////////////////////////////////////////////////////////////////////////
  void exportTreeStructure(std::vector<std::vector<IndexType>> &tree) {
    if (level() >= tree.size()) tree.resize(level() + 1);
    tree[level()].push_back(node().indices_.size());
    for (auto i = 0; i < nSons(); ++i) sons(i).exportTreeStructure(tree);
  }
  //////////////////////////////////////////////////////////////////////////////
  void getLeafIterator(std::vector<const ClusterTree *> &leafs) const {
    if (nSons() == 0)
      leafs.push_back(this);
    else
      for (auto i = 0; i < nSons(); ++i) sons(i).getLeafIterator(leafs);
    return;
  }

 private:
  //////////////////////////////////////////////////////////////////////////////
  // private methods
  //////////////////////////////////////////////////////////////////////////////
  void initBoundingBox(const eigenMatrix &P) {
    node().bb_.resize(P.rows(), 3);
    node().bb_.col(0) = P.rowwise().minCoeff();
    node().bb_.col(1) = P.rowwise().maxCoeff();
    node().bb_.col(0) += FMCA_BBOX_THREASHOLD * node().bb_.col(0);
    node().bb_.col(1) += FMCA_BBOX_THREASHOLD * node().bb_.col(1);
    node().bb_.col(2) = node().bb_.col(1) - node().bb_.col(0);
    return;
  }
  // recursively perform clustering on sons
  void computeClusters(const eigenMatrix &P, IndexType min_cluster_size) {
    Splitter split;
    node().tree_data_->max_level_ = node().tree_data_->max_level_ < level()
                                       ? level()
                                       : node().tree_data_->max_level_;
    if (node().indices_.size() > min_cluster_size) {
      appendSons(2);
      // set up bounding boxes for sons
      for (auto i = 0; i < 2; ++i) {
        sons(i).node().bb_ = node().bb_;
        sons(i).node().tree_data_ = node().tree_data_;
        sons(i).node().indices_begin_ = node().indices_begin_;
      }
      // split index set and set sons bounding boxes
      split(P, node().indices_, node().bb_, sons(0).node(), sons(1).node());
      // let recursion handle the rest
      for (auto i = 0; i < nSons(); ++i)
        sons(i).computeClusters(P, min_cluster_size);
      // make indices hierarchically
      node().indices_.clear();
      for (auto i = 0; i < nSons(); ++i)
        node().indices_.insert(node().indices_.end(),
                               sons(i).node().indices_.begin(),
                               sons(i).node().indices_.end());
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // make bounding boxes tight
  //////////////////////////////////////////////////////////////////////////////
  void shrinkToFit(const eigenMatrix &P) {
    eigenMatrix bbmat(P.rows(), 3);
    if (nSons()) {
      // assert that all sons have fitted bb's
      for (auto i = 0; i < nSons(); ++i) sons(i).shrinkToFit(P);
      // now update own bb (we need a son with indices to get a first bb)
      for (auto i = 0; i < nSons(); ++i)
        if (sons(i).node().indices_.size()) {
          bbmat.col(0).array() = sons(i).node().bb_.col(0);
          bbmat.col(1).array() = sons(i).node().bb_.col(1);
          break;
        }
      for (auto i = 0; i < nSons(); ++i)
        if (sons(i).node().indices_.size()) {
          bbmat.col(0).array() =
              bbmat.col(0).array().min(sons(i).node().bb_.col(0).array());
          bbmat.col(1).array() =
              bbmat.col(1).array().max(sons(i).node().bb_.col(1).array());
        }
    } else {
      if (node().indices_.size()) {
        for (auto j = 0; j < P.rows(); ++j) {
          bbmat(j, 0) = P(j, node().indices_[0]);
          bbmat(j, 1) = P(j, node().indices_[0]);
        }
        for (auto i = 1; i < node().indices_.size(); ++i)
          for (auto j = 0; j < P.rows(); ++j) {
            // determine minimum
            bbmat(j, 0) = bbmat(j, 0) <= P(j, node().indices_[i])
                              ? bbmat(j, 0)
                              : P(j, node().indices_[i]);
            // determine maximum
            bbmat(j, 1) = bbmat(j, 1) >= P(j, node().indices_[i])
                              ? bbmat(j, 1)
                              : P(j, node().indices_[i]);
          }
        bbmat.col(0).array() -= 10 * FMCA_ZERO_TOLERANCE;
        bbmat.col(1).array() += 10 * FMCA_ZERO_TOLERANCE;
      } else {
        // collapse empty box to its midpoint
        bbmat.col(0) = 0.5 * (node().bb_.col(0) + node().bb_.col(1));
        bbmat.col(1) = bbmat.col(0);
      }
    }
    bbmat.col(2) = bbmat.col(1) - bbmat.col(0);
    node().bb_ = bbmat;
    return;
  }
};

}  // namespace FMCA
#endif

// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
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

template <typename ValueType, IndexType Dim> struct ClusterTreeData {
  ValueType geometry_diam_ = 0;
  IndexType max_id_ = 0;
  IndexType max_level_ = 0;
};

/**
 *  \ingroup ClusterTree
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
template <typename ValueType, IndexType Dim, IndexType MinClusterSize,
          typename Splitter =
              ClusterSplitter::CardinalityBisection<ValueType, Dim>>
class ClusterTree {
  friend Splitter;

public:
  typedef ValueType value_type;
  enum { dimension = Dim };
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTree() {}
  ClusterTree(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    init(P);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    // set up bounding box for root node
    tree_data_ = std::make_shared<ClusterTreeData<ValueType, Dim>>();
    initBoundingBox(P);
    level_ = 0;
    id_ = 0;
    indices_begin_ = 0;
    indices_.resize(P.cols());
    std::iota(std::begin(indices_), std::end(indices_), 0u);
    computeClusters(P);
    shrinkToFit(P);
    tree_data_->geometry_diam_ = bb_.col(2).norm();
  }
  //////////////////////////////////////////////////////////////////////////////
  // get a vector with the bounding boxes on a certain level
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVector(std::vector<Eigen::Matrix<ValueType, Dim, 3u>> *bbvec,
                      IndexType level = 0) {
    if (sons_.size() && level_ < level)
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].get_BboxVector(bbvec, level);
    if (level_ == level)
      if (indices_.size())
        bbvec->push_back(bb_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void
  get_BboxVectorLeafs(std::vector<Eigen::Matrix<ValueType, Dim, 3u>> *bbvec) {
    if (sons_.size())
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].get_BboxVectorLeafs(bbvec);
    else if (indices_.size())
      bbvec->push_back(bb_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // getter
  //////////////////////////////////////////////////////////////////////////////
  const ClusterTree *get_cluster() const { return this; }

  const Eigen::Matrix<ValueType, Dim, 3u> &get_bb() const { return bb_; }

  const std::vector<IndexType> &get_indices() const { return indices_; }

  const std::vector<ClusterTree> &get_sons() const { return sons_; }

  const ClusterTreeData<ValueType, Dim> &get_tree_data() const {
    return *tree_data_;
  }

  IndexType get_level() const { return level_; }

  IndexType get_id() const { return id_; }

  IndexType get_indices_begin() const { return indices_begin_; }
  //////////////////////////////////////////////////////////////////////////////
  void exportTreeStructure(std::vector<std::vector<IndexType>> &tree) {
    if (level_ >= tree.size())
      tree.resize(level_ + 1);
    tree[level_].push_back(indices_.size());
    for (auto i = 0; i < sons_.size(); ++i)
      sons_[i].exportTreeStructure(tree);
  }
  //////////////////////////////////////////////////////////////////////////////
  void getLeafIterator(std::vector<ClusterTree *> &leafs) {
    if (sons_.size() == 0)
      leafs.push_back(this);
    else
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].getLeafIterator(leafs);
  }
  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  // private methods
  //////////////////////////////////////////////////////////////////////////////
  void initBoundingBox(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    for (auto i = 0; i < Dim; ++i) {
      bb_(i, 0) = P.row(i).minCoeff();
      bb_(i, 1) = P.row(i).maxCoeff();
      // add some padding, e.g. 5%, see FMCA_BBOX_THREASHOLD
      bb_(i, 0) = bb_(i, 0) < 0 ? bb_(i, 0) * (1 + FMCA_BBOX_THREASHOLD)
                                : bb_(i, 0) * (1 - FMCA_BBOX_THREASHOLD);
      bb_(i, 1) = bb_(i, 1) < 0 ? bb_(i, 1) * (1 - FMCA_BBOX_THREASHOLD)
                                : bb_(i, 1) * (1 + FMCA_BBOX_THREASHOLD);
    }
    bb_.col(2) = bb_.col(1) - bb_.col(0);
    return;
  }

  // recursively perform clustering on sons
  void computeClusters(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    Splitter split;
    if (indices_.size() > MinClusterSize) {
      sons_.resize(2);
      // set up bounding boxes for sons (which are traversed in a binary
      // fashion)
      for (auto i = 0; i < 2; ++i) {
        sons_[i].level_ = level_ + 1;
        tree_data_->max_level_ = (tree_data_->max_level_ < level_ + 1)
                                     ? (level_ + 1)
                                     : tree_data_->max_level_;
        sons_[i].id_ = (1 << (level_ + 1)) + 2 * id_ + i;
        tree_data_->max_id_ = tree_data_->max_id_ < sons_[i].id_
                                  ? sons_[i].id_
                                  : tree_data_->max_id_;
        sons_[i].bb_ = bb_;
        sons_[i].tree_data_ = tree_data_;
        sons_[i].indices_begin_ = indices_begin_;
      }
      // split index set and set sons bounding boxes
      split(P, indices_, bb_, sons_[0], sons_[1]);
      // let recursion handle the rest
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].computeClusters(P);
      // make indices hierarchically
      indices_.clear();
      for (auto i = 0; i < sons_.size(); ++i)
        indices_.insert(indices_.end(), sons_[i].indices_.begin(),
                        sons_[i].indices_.end());
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // make bounding boxes tight
  //////////////////////////////////////////////////////////////////////////////
  void shrinkToFit(const Eigen::Matrix<ValueType, Dim, Eigen::Dynamic> &P) {
    Eigen::Matrix<ValueType, Dim, 3u> bbmat =
        Eigen::Matrix<ValueType, Dim, 3u>::Zero();
    if (sons_.size()) {
      // assert that all sons have fitted bb's
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].shrinkToFit(P);
      // now update own bb
      // we need a son with indices to get a first bb
      for (auto i = 0; i < sons_.size(); ++i)
        if (sons_[i].indices_.size()) {
          bbmat.col(0).array() = sons_[i].bb_.col(0);
          bbmat.col(1).array() = sons_[i].bb_.col(1);
          break;
        }
      for (auto i = 0; i < sons_.size(); ++i)
        if (sons_[i].indices_.size()) {
          bbmat.col(0).array() =
              bbmat.col(0).array().min(sons_[i].bb_.col(0).array());
          bbmat.col(1).array() =
              bbmat.col(1).array().max(sons_[i].bb_.col(1).array());
        }
    } else {
      if (indices_.size()) {
        for (auto j = 0; j < Dim; ++j) {
          bbmat(j, 0) = P(j, indices_[0]);
          bbmat(j, 1) = P(j, indices_[0]);
        }
        for (auto i = 1; i < indices_.size(); ++i)
          for (auto j = 0; j < Dim; ++j) {
            // determine minimum
            bbmat(j, 0) = bbmat(j, 0) <= P(j, indices_[i]) ? bbmat(j, 0)
                                                           : P(j, indices_[i]);
            // determine maximum
            bbmat(j, 1) = bbmat(j, 1) >= P(j, indices_[i]) ? bbmat(j, 1)
                                                           : P(j, indices_[i]);
          }
        bbmat.col(0).array() -= 10 * FMCA_ZERO_TOLERANCE;
        bbmat.col(1).array() += 10 * FMCA_ZERO_TOLERANCE;
      } else {
        // collapse empty box to its midpoint
        bbmat.col(0) = 0.5 * (bb_.col(0) + bb_.col(1));
        bbmat.col(1) = bbmat.col(0);
      }
    }
    bbmat.col(2) = bbmat.col(1) - bbmat.col(0);
    bb_ = bbmat;
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // private member variables
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<ValueType, Dim, 3u> bb_;
  std::vector<IndexType> indices_;
  std::vector<ClusterTree> sons_;
  std::shared_ptr<ClusterTreeData<ValueType, Dim>> tree_data_;
  IndexType level_;
  IndexType indices_begin_;
  IndexType id_;
};
} // namespace FMCA
#endif

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

struct ClusterTreeData {
  double geometry_diam_;
};

/**
 *  \ingroup ClusterTree
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
template <typename T, unsigned int Dim, unsigned int MinClusterSize,
          typename Splitter = ClusterSplitter::CardinalityBisection<T, Dim>>
class ClusterTree {
  friend Splitter;

public:
  typedef T value_type;
  enum { dimension = Dim };
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTree() {}
  ClusterTree(const Eigen::Matrix<T, Dim, Eigen::Dynamic> &P) { init(P); }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Eigen::Matrix<T, Dim, Eigen::Dynamic> &P) {
    // set up bounding box for root node
    initBoundingBox(P);
    tree_data_ = std::make_shared<ClusterTreeData>();
    tree_data_->geometry_diam_ = bb_.col(2).norm();
    level_ = 0;
    id_ = 0;
    indices_.resize(P.cols());
    std::iota(std::begin(indices_), std::end(indices_), 0u);
    computeClusters(P);
    shrinkToFit(P);
  }
  //////////////////////////////////////////////////////////////////////////////
  // get a vector with the bounding boxes on a certain level
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVector(std::vector<Eigen::Matrix<T, Dim, 3u>> *bbvec,
                      unsigned int level = 0) {
    if (sons_.size() && level_ < level)
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].get_BboxVector(bbvec, level);
    if (level_ == level)
      if (indices_.size())
        bbvec->push_back(bb_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVectorLeafs(std::vector<Eigen::Matrix<T, Dim, 3u>> *bbvec) {
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

  const Eigen::Matrix<T, Dim, 3u> &get_bb() const { return bb_; }

  const std::vector<unsigned int> &get_indices() const { return indices_; }

  const std::vector<ClusterTree> &get_sons() const { return sons_; }

  const ClusterTreeData &get_tree_data() const { return *tree_data_; }

  unsigned int get_level() const { return level_; }

  unsigned int get_id() const { return id_; }
  //////////////////////////////////////////////////////////////////////////////
  void exportTreeStructure(std::vector<std::vector<int>> &tree) {
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
  void initBoundingBox(const Eigen::Matrix<T, Dim, Eigen::Dynamic> &P) {
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
  void computeClusters(const Eigen::Matrix<T, Dim, Eigen::Dynamic> &P) {
    Splitter split;
    if (indices_.size() > MinClusterSize) {
      sons_.resize(2);
      // set up bounding boxes for sons (which are traversed in a binary
      // fashion)
      for (auto i = 0; i < 2; ++i) {
        sons_[i].level_ = level_ + 1;
        sons_[i].id_ = 2 * id_ + i;
        sons_[i].bb_ = bb_;
        sons_[i].tree_data_ = tree_data_;
      }
      // split index set and set sons bounding boxes
      split(P, indices_, bb_, sons_[0], sons_[1]);
      // let recursion handle the rest
      for (auto i = 0; i < sons_.size(); ++i)
        sons_[i].computeClusters(P);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // make bounding boxes tight
  //////////////////////////////////////////////////////////////////////////////
  void shrinkToFit(const Eigen::Matrix<T, Dim, Eigen::Dynamic> &P) {
    Eigen::Matrix<T, Dim, 3u> bbmat = Eigen::Matrix<T, Dim, 3u>::Zero();
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
  Eigen::Matrix<T, Dim, 3u> bb_;
  std::vector<unsigned int> indices_;
  std::vector<ClusterTree> sons_;
  std::shared_ptr<ClusterTreeData> tree_data_;
  unsigned int level_;
  unsigned int id_;
}; // namespace FMCA
} // namespace FMCA
#endif

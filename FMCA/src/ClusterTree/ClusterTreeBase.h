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
#ifndef FMCA_CLUSTERTREE_CLUSTERTREEBASE_H_
#define FMCA_CLUSTERTREE_CLUSTERTREEBASE_H_

namespace FMCA {

/**
 *  \ingroup ClusterTree
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
template <typename Derived> class ClusterTreeBase : public TreeBase<Derived> {
public:
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  typedef typename internal::traits<Derived>::node_type node_type;
  typedef typename internal::traits<Derived>::Splitter Splitter;
  // make base class methods visible
  using TreeBase<Derived>::node;
  using TreeBase<Derived>::sons;
  using TreeBase<Derived>::appendSons;
  using TreeBase<Derived>::nSons;
  using TreeBase<Derived>::level;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTreeBase() {}
  ClusterTreeBase(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    init(P, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  void init(const eigenMatrix &P, IndexType min_cluster_size = 1) {
    // set up bounding box for root node
    initBoundingBox(P);
    node().indices_begin_ = 0;
    node().indices_.resize(P.cols());
    std::iota(node().indices_.begin(), node().indices_.end(), 0u);
    computeClusters(P, min_cluster_size);
    shrinkToFit(P);
  }
  //////////////////////////////////////////////////////////////////////////////
  // getter
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &bb() const { return node().bb_; }

  const std::vector<IndexType> &indices() const { return node().indices_; }

  IndexType indices_begin() const { return node().indices_begin_; }

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
    if (node().indices_.size() > min_cluster_size) {
      appendSons(2);
      // set up bounding boxes for sons
      for (auto i = 0; i < 2; ++i) {
        sons(i).node().bb_ = node().bb_;
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
      for (auto i = 0; i < nSons(); ++i)
        sons(i).shrinkToFit(P);
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

} // namespace FMCA
#endif

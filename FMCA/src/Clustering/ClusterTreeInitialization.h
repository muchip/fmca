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
#ifndef FMCA_CLUSTERING_INITCLUSTERTREEIMPL_H_
#define FMCA_CLUSTERING_INITCLUSTERTREEIMPL_H_

namespace FMCA {
namespace internal {

template <typename Derived, typename eigenMatrix>
void init_BoundingBox_impl(ClusterTreeBase<Derived> &CT, const eigenMatrix &P,
                           IndexType min_cluster_size) {
  CT.node().bb_.resize(P.rows(), 3);
  CT.node().bb_.col(0) = P.rowwise().minCoeff();
  CT.node().bb_.col(1) = P.rowwise().maxCoeff();
  CT.node().bb_.col(0) += FMCA_BBOX_THREASHOLD * CT.node().bb_.col(0);
  CT.node().bb_.col(1) += FMCA_BBOX_THREASHOLD * CT.node().bb_.col(1);
  CT.node().bb_.col(2) = CT.node().bb_.col(1) - CT.node().bb_.col(0);
  return;
}

template <typename Derived, typename eigenMatrix>
void init_ClusterTree_impl(ClusterTreeBase<Derived> &CT, const eigenMatrix &P,
                           IndexType min_cluster_size) {
  typename traits<Derived>::Splitter split;
  if (CT.node().indices_.size() > min_cluster_size) {
    CT.appendSons(2);
    // set up bounding boxes for sons
    for (auto i = 0; i < 2; ++i) {
      CT.sons(i).node().bb_ = CT.node().bb_;
      CT.sons(i).node().indices_begin_ = CT.node().indices_begin_;
    }
    // split index set and set sons bounding boxes
    split(P, CT.node().indices_, CT.node().bb_, CT.sons(0).node(),
          CT.sons(1).node());
    // let recursion handle the rest
    for (auto i = 0; i < CT.nSons(); ++i)
      init_ClusterTree_impl<Derived, eigenMatrix>(CT.sons(i), P,
                                                  min_cluster_size);
    // make indices hierarchically
    CT.node().indices_.clear();
    for (auto i = 0; i < CT.nSons(); ++i)
      CT.node().indices_.insert(CT.node().indices_.end(),
                                CT.sons(i).node().indices_.begin(),
                                CT.sons(i).node().indices_.end());
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// make bounding boxes tight
//////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename eigenMatrix>
void shrinkToFit_impl(ClusterTreeBase<Derived> &CT, const eigenMatrix &P) {
  eigenMatrix bbmat(P.rows(), 3);
  if (CT.nSons()) {
    // assert that all sons have fitted bb's
    for (auto i = 0; i < CT.nSons(); ++i) shrinkToFit_impl(CT.sons(i), P);
    // now update own bb (we need a son with indices to get a first bb)
    for (auto i = 0; i < CT.nSons(); ++i)
      if (CT.sons(i).node().indices_.size()) {
        bbmat.col(0).array() = CT.sons(i).node().bb_.col(0);
        bbmat.col(1).array() = CT.sons(i).node().bb_.col(1);
        break;
      }
    for (auto i = 0; i < CT.nSons(); ++i)
      if (CT.sons(i).node().indices_.size()) {
        bbmat.col(0).array() =
            bbmat.col(0).array().min(CT.sons(i).node().bb_.col(0).array());
        bbmat.col(1).array() =
            bbmat.col(1).array().max(CT.sons(i).node().bb_.col(1).array());
      }
  } else {
    if (CT.node().indices_.size()) {
      for (auto j = 0; j < P.rows(); ++j) {
        bbmat(j, 0) = P(j, CT.node().indices_[0]);
        bbmat(j, 1) = P(j, CT.node().indices_[0]);
      }
      for (auto i = 1; i < CT.node().indices_.size(); ++i)
        for (auto j = 0; j < P.rows(); ++j) {
          // determine minimum
          bbmat(j, 0) = bbmat(j, 0) <= P(j, CT.node().indices_[i])
                            ? bbmat(j, 0)
                            : P(j, CT.node().indices_[i]);
          // determine maximum
          bbmat(j, 1) = bbmat(j, 1) >= P(j, CT.node().indices_[i])
                            ? bbmat(j, 1)
                            : P(j, CT.node().indices_[i]);
        }
      bbmat.col(0).array() -= 10 * FMCA_ZERO_TOLERANCE;
      bbmat.col(1).array() += 10 * FMCA_ZERO_TOLERANCE;
    } else {
      // collapse empty box to its midpoint
      bbmat.col(0) = 0.5 * (CT.node().bb_.col(0) + CT.node().bb_.col(1));
      bbmat.col(1) = bbmat.col(0);
    }
  }
  bbmat.col(2) = bbmat.col(1) - bbmat.col(0);
  CT.node().bb_ = bbmat;
  return;
}

}  // namespace internal

}  // namespace FMCA

#endif

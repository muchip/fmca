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
/** \ingroup internal
 *  \brief initializes a bounding box for the geometry
 **/
template <>
struct ClusterTreeInitializer<ClusterTree> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename eigenMatrix>
  static void init(ClusterTreeBase<Derived> &CT, IndexType min_cluster_size,
                   const eigenMatrix &P) {
    init_BoundingBox_impl(CT, min_cluster_size, P);
    CT.node().indices_begin_ = 0;
    CT.node().indices_.resize(P.cols());
    std::iota(CT.node().indices_.begin(), CT.node().indices_.end(), 0u);
    init_ClusterTree_impl(CT, min_cluster_size, P);
    shrinkToFit_impl(CT, P);
    IndexType i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename eigenMatrix>
  static void init_BoundingBox_impl(ClusterTreeBase<Derived> &CT,
                                    IndexType min_cluster_size,
                                    const eigenMatrix &P) {
    CT.node().bb_.resize(P.rows(), 3);
    CT.node().bb_.col(0) = P.rowwise().minCoeff();
    CT.node().bb_.col(1) = P.rowwise().maxCoeff();
    CT.node().bb_.col(0) += FMCA_BBOX_THREASHOLD * CT.node().bb_.col(0);
    CT.node().bb_.col(1) += FMCA_BBOX_THREASHOLD * CT.node().bb_.col(1);
    CT.node().bb_.col(2) = CT.node().bb_.col(1) - CT.node().bb_.col(0);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /** \ingroup internal
   *  \brief perform cluster refinement given a Splitter class
   **/
  template <typename Derived, typename eigenMatrix>
  static void init_ClusterTree_impl(ClusterTreeBase<Derived> &CT,
                                    IndexType min_cluster_size,
                                    const eigenMatrix &P) {
    typename traits<Derived>::Splitter split;
    const IndexType split_threshold =
        min_cluster_size >= 1 ? (2 * min_cluster_size - 1) : 1;
    if (CT.node().indices_.size() > split_threshold) {
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
        init_ClusterTree_impl<Derived, eigenMatrix>(CT.sons(i),
                                                    min_cluster_size, P);
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
  /** \ingroup internal
   *  \brief recursively shrink all bounding boxes to the minimal possible
   *         size
   **/
  template <typename Derived, typename eigenMatrix>
  static void shrinkToFit_impl(ClusterTreeBase<Derived> &CT,
                               const eigenMatrix &P) {
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
};
}  // namespace internal

}  // namespace FMCA

#endif

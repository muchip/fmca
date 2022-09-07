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
  template <typename Derived>
  static void init(ClusterTreeBase<Derived> &CT, Index min_csize,
                   const Matrix &P) {
    init_BoundingBox_impl(CT, min_csize, P);
    CT.node().indices_begin_ = 0;
    CT.node().indices_.resize(P.cols());
    std::iota(CT.node().indices_.begin(), CT.node().indices_.end(), 0u);
    init_ClusterTree_impl(CT, min_csize, P);
    shrinkToFit_impl(CT, P);
    Index i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  static void init_BoundingBox_impl(ClusterTreeBase<Derived> &CT,
                                    Index min_csize, const Matrix &P) {
    CT.node().bb_.resize(P.rows(), 3);
    CT.node().bb_.col(0) = P.rowwise().minCoeff();
    CT.node().bb_.col(1) = P.rowwise().maxCoeff();
    // increase bounding box by an epsilon layer
    CT.node().bb_.col(0) -=
        FMCA_BBOX_THREASHOLD * CT.node().bb_.col(0).cwiseAbs();
    CT.node().bb_.col(1) +=
        FMCA_BBOX_THREASHOLD * CT.node().bb_.col(1).cwiseAbs();
    CT.node().bb_.col(2) = CT.node().bb_.col(1) - CT.node().bb_.col(0);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /** \ingroup internal
   *  \brief perform cluster refinement given a Splitter class
   **/
  template <typename Derived>
  static void init_ClusterTree_impl(ClusterTreeBase<Derived> &CT,
                                    Index min_csize, const Matrix &P) {
    typename traits<Derived>::Splitter split;
    const Index split_threshold = min_csize >= 1 ? (2 * min_csize - 1) : 1;
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
        init_ClusterTree_impl<Derived>(CT.sons(i), min_csize, P);
      // add sorted indices to father cluster
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
  template <typename Derived>
  static void shrinkToFit_impl(ClusterTreeBase<Derived> &CT, const Matrix &P) {
    Matrix bbmat(P.rows(), 3);
    if (CT.node().indices_.size()) {
      bbmat.col(0) = P.col(CT.node().indices_[0]);
      bbmat.col(1) = bbmat.col(0);
      if (CT.nSons()) {
        // assert that all sons have fitted bb's
        for (auto i = 0; i < CT.nSons(); ++i) {
          shrinkToFit_impl(CT.sons(i), P);
          if (CT.sons(i).node().indices_.size()) {
            bbmat.col(0).array() =
                bbmat.col(0).array().min(CT.sons(i).node().bb_.col(0).array());
            bbmat.col(1).array() =
                bbmat.col(1).array().max(CT.sons(i).node().bb_.col(1).array());
          }
        }
      } else {
        for (auto i = 0; i < CT.node().indices_.size(); ++i) {
          // determine minimum
          bbmat.col(0).array() =
              P.col(CT.node().indices_[i]).array().min(bbmat.col(0).array());
          // determine maximum
          bbmat.col(1).array() =
              P.col(CT.node().indices_[i]).array().max(bbmat.col(1).array());
        }
      }
    } else {
      // set everything to inf;
      bbmat.setOnes();
      bbmat.col(0) *= FMCA_INF;
      bbmat.col(1) *= -FMCA_INF;
    }
    bbmat.col(2) = bbmat.col(1) - bbmat.col(0);
    CT.node().bb_ = bbmat;
    return;
  }
};
}  // namespace internal

}  // namespace FMCA

#endif

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
#ifndef FMCA_CLUSTERING_INITCLUSTERTREEGRAPHIMPL_H_
#define FMCA_CLUSTERING_INITCLUSTERTREEGRAPHIMPL_H_

namespace FMCA {
namespace internal {
/** \ingroup internal
 *  \brief initializes a bounding box for the geometry
 **/
template <>
struct ClusterTreeInitializer<ClusterTreeGraph> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename Derived2, typename Derived3, typename logicType>
  static void init(ClusterTreeBase<Derived> &CT, IndexType min_cluster_size,
                   const Eigen::MatrixBase<Derived2> &V,
                   const Eigen::MatrixBase<Derived3> &F, const logicType dual) {
    typedef typename ClusterTreeBase<Derived>::eigenMatrix eigenMatrix;
    // we split the graph by using metis
    // midpoints
    eigenMatrix P(V.cols(), F.rows());
    P.setZero();
    for (auto i = 0; i < P.cols(); ++i)
      for (auto j = 0; j < F.cols(); ++j)
        P.col(i) += V.row(F(i, j)).transpose() / F.cols();
    if (dual) {
      ClusterTreeInitializer<ClusterTree>::init_BoundingBox_impl(
          CT, min_cluster_size, P);
      CT.node().indices_begin_ = 0;
      CT.node().indices_.resize(P.cols());
    }

    else {
      ClusterTreeInitializer<ClusterTree>::init_BoundingBox_impl(
          CT, min_cluster_size, V.transpose());
      CT.node().indices_begin_ = 0;
      CT.node().indices_.resize(V.rows());
    }

    std::iota(CT.node().indices_.begin(), CT.node().indices_.end(), 0u);
    init_ClusterTreeGraph_impl(CT, min_cluster_size, V, F, dual);
    //if (dual)
        //ClusterTreeInitializer<ClusterTreeMesh>::shrinkToFit_impl(CT, V, F);
    IndexType i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  /** \ingroup internal
   *  \brief perform cluster refinement given a Splitter class
   **/
  template <typename Derived, typename Derived2, typename Derived3, typename logicType>
  static void init_ClusterTreeGraph_impl(ClusterTreeBase<Derived> &CT,
                                         IndexType min_cluster_size,
                                         const Eigen::MatrixBase<Derived2> &V,
                                         const Eigen::MatrixBase<Derived3> &F,
                                         const logicType dual) {
    typename traits<Derived>::Splitter split;
    if (CT.node().indices_.size() > 2 * min_cluster_size) {
      CT.appendSons(2);
      // set up bounding boxes for sons
      for (auto i = 0; i < 2; ++i) {
        CT.sons(i).node().bb_ = CT.node().bb_;
        CT.sons(i).node().indices_begin_ = CT.node().indices_begin_;
      }
      // split index set and set sons bounding boxes
      split(V, F, CT.node().indices_, CT.node().bb_, CT.sons(0).node(),
            CT.sons(1).node(), dual);
      // let recursion handle the rest
      for (auto i = 0; i < CT.nSons(); ++i)
        init_ClusterTreeGraph_impl<Derived, Derived2, Derived3, logicType>(
            CT.sons(i), min_cluster_size, V, F, dual);
      // make indices hierarchically
      CT.node().indices_.clear();
      for (auto i = 0; i < CT.nSons(); ++i)
        CT.node().indices_.insert(CT.node().indices_.end(),
                                  CT.sons(i).node().indices_.begin(),
                                  CT.sons(i).node().indices_.end());
    }
    return;
  }
};
}  // namespace internal

}  // namespace FMCA

#endif

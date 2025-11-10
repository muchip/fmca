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
#ifndef FMCA_CLUSTERING_INITCLUSTERTREEMESHIMPL_H_
#define FMCA_CLUSTERING_INITCLUSTERTREEMESHIMPL_H_

namespace FMCA {
namespace internal {
/** \ingroup internal
 *  \brief initializes a bounding box for the geometry
 **/
template <>
struct ClusterTreeInitializer<ClusterTreeMesh> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename Derived2, typename Derived3>
  static void init(ClusterTreeBase<Derived> &CT, Index min_cluster_size,
                   const MatrixBase<Derived2> &V,
                   const MatrixBase<Derived3> &F) {
    // we split according to midpoints and fix everything later
    Matrix P(V.cols(), F.rows());
    P.setZero();
    for (auto i = 0; i < P.cols(); ++i)
      for (auto j = 0; j < F.cols(); ++j)
        P.col(i) += V.row(F(i, j)).transpose() / F.cols();
    ClusterTreeInitializer<ClusterTree>::init_BoundingBox_impl(
        CT, min_cluster_size, P);
    CT.node().indices_begin_ = 0;
    CT.node().indices_ = std::shared_ptr<Index>(new Index[P.cols()],
                                                std::default_delete<Index[]>());
    CT.node().block_size_ = P.cols();
    Index *indices = CT.node().indices_.get();
    for (Index i = 0; i < CT.block_size(); ++i) indices[i] = i;
    ClusterTreeInitializer<ClusterTree>::init_ClusterTree_impl(
        CT, min_cluster_size, P);
    shrinkToFit_impl(CT, V, F);
    Index i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  /** \ingroup internal
   *  \brief recursively shrink all bounding boxes to the minimal possible
   *         size
   **/
  template <typename Derived, typename Derived2, typename Derived3>
  static void shrinkToFit_impl(ClusterTreeBase<Derived> &CT,
                               const MatrixBase<Derived2> &V,
                               const MatrixBase<Derived3> &F) {
    Matrix bbmat(V.cols(), 3);
    if (CT.nSons()) {
      // assert that all sons have fitted bb's
      for (auto i = 0; i < CT.nSons(); ++i) shrinkToFit_impl(CT.sons(i), V, F);
      // now update own bb (we need a son with indices to get a first bb)
      for (auto i = 0; i < CT.nSons(); ++i)
        if (CT.sons(i).block_size()) {
          bbmat.col(0).array() = CT.sons(i).node().bb_.col(0);
          bbmat.col(1).array() = CT.sons(i).node().bb_.col(1);
          break;
        }
      for (auto i = 0; i < CT.nSons(); ++i)
        if (CT.sons(i).block_size()) {
          bbmat.col(0).array() =
              bbmat.col(0).array().min(CT.sons(i).node().bb_.col(0).array());
          bbmat.col(1).array() =
              bbmat.col(1).array().max(CT.sons(i).node().bb_.col(1).array());
        }
    } else {
      // this is the major change compared to the other boxes.
      // Here, we need to make sure that all vertices of an element are
      // contained in the box
      if (CT.block_size()) {
        bbmat.col(0) = V.row(F(CT.indices()[0], 0)).transpose();
        bbmat.col(1) = bbmat.col(0);

        for (auto i = 0; i < CT.block_size(); ++i)
          for (auto k = 0; k < F.cols(); ++k) {
            bbmat.col(0).array() = bbmat.col(0).array().min(
                V.row(F(CT.indices()[i], k)).transpose().array());
            bbmat.col(1).array() = bbmat.col(1).array().max(
                V.row(F(CT.indices()[i], k)).transpose().array());
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

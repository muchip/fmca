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
template <> struct ClusterTreeInitializer<ClusterTreeMesh> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  static void init(ClusterTreeBase<Derived> &CT, Index min_csize,
                   const Matrix &V, const iMatrix &F) {
    typedef ClusterTreeInitializer<ClusterTree> CTInitializer;
    // we split according to midpoints and fix everything later
    Matrix P(V.cols(), F.rows());
    P.setZero();
    for (auto i = 0; i < P.cols(); ++i)
      for (auto j = 0; j < F.cols(); ++j)
        P.col(i) += V.row(F(i, j)).transpose() / F.cols();
    CTInitializer::init_BoundingBox_impl(CT, min_csize, P);
    CT.node().indices_begin_ = 0;
    CT.node().indices_.resize(P.cols());
    std::iota(CT.node().indices_.begin(), CT.node().indices_.end(), 0u);
    CTInitializer::init_ClusterTree_impl(CT, min_csize, P);
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
  template <typename Derived>
  static void shrinkToFit_impl(ClusterTreeBase<Derived> &CT, const Matrix &V,
                               const iMatrix &F) {
    Matrix bbmat(V.cols(), 3);
    if (CT.nSons()) {
      // assert that all sons have fitted bb's
      for (auto i = 0; i < CT.nSons(); ++i)
        shrinkToFit_impl(CT.sons(i), V, F);
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
      // this is the major change compared to the other boxes.
      // Here, we need to make sure that all vertices of an element are
      // contained in the box
      if (CT.node().indices_.size()) {
        bbmat.col(0) = V.row(F(CT.node().indices_[0], 0)).transpose();
        bbmat.col(1) = bbmat.col(0);

        for (auto i = 0; i < CT.node().indices_.size(); ++i)
          for (auto k = 0; k < F.cols(); ++k) {
            bbmat.col(0).array() = bbmat.col(0).array().min(
                V.row(F(CT.node().indices_[i], k)).transpose().array());
            bbmat.col(1).array() = bbmat.col(1).array().max(
                V.row(F(CT.node().indices_[i], k)).transpose().array());
          }
      } else {
        // set everything to inf;
        bbmat.setOnes();
        bbmat.col(0) *= Scalar(1. / 0.);
        bbmat.col(1) *= -Scalar(1. / 0.);
      }
    }
    bbmat.col(2) = bbmat.col(1) - bbmat.col(0);
    CT.node().bb_ = bbmat;
    return;
  }
};
} // namespace internal

} // namespace FMCA

#endif

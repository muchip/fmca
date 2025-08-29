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
#ifndef FMCA_CLUSTERING_INITSPHERECLUSTERTREEIMPL_H_
#define FMCA_CLUSTERING_INITSPHERECLUSTERTREEIMPL_H_

namespace FMCA {
namespace internal {
/** \ingroup internal
 *  \brief initializes a bounding box for the geometry
 **/
template <>
struct ClusterTreeInitializer<SphereClusterTree> {
  ClusterTreeInitializer() = delete;
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  static void init(ClusterTreeBase<Derived> &CT, Index min_csize,
                   const Matrix &P) {
    init_BoundingBox_impl(CT, min_csize, P);
    CT.node().indices_begin_ = 0;
    CT.node().indices_ = std::shared_ptr<Index>(new Index[P.cols()],
                                                std::default_delete<Index[]>());
    CT.node().block_size_ = P.cols();
    Index *indices = CT.node().indices_.get();
    for (Index i = 0; i < CT.block_size(); ++i) indices[i] = i;
    init_ClusterTree_impl(CT, min_csize, P);
    shrinkToFit_impl(CT, P);
    Index i = 0;
    for (auto &it : CT) {
      it.node().block_id_ = i;
      ++i;
    }
    computeCR(P, CT);
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
    const Index K = 1 << P.rows();
    const Index split_threshold = min_csize >= 1 ? (K * min_csize - 1) : 1;
    if (CT.node().block_size_ > split_threshold) {
      CT.appendSons(K);
      // set up bounding boxes for sons
      for (Index i = 0; i < K; ++i) {
        CT.sons(i).node().bb_ = CT.node().bb_;
        CT.sons(i).node().indices_ = CT.node().indices_;
        CT.sons(i).node().block_size_ = CT.node().block_size_;
        CT.sons(i).node().indices_begin_ = CT.node().indices_begin_;
      }
      // split index set and set sons bounding boxes
      split(P, CT.sons_vector());
      // let recursion handle the rest
      for (Index i = 0; i < CT.nSons(); ++i)
        init_ClusterTree_impl<Derived>(CT.sons(i), min_csize, P);
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
    if (CT.block_size()) {
      bbmat.col(0) = P.col(CT.indices()[0]);
      bbmat.col(1) = bbmat.col(0);
      if (CT.nSons()) {
        // assert that all sons have fitted bb's
        for (Index i = 0; i < CT.nSons(); ++i) {
          shrinkToFit_impl(CT.sons(i), P);
          if (CT.sons(i).block_size()) {
            bbmat.col(0).array() =
                bbmat.col(0).array().min(CT.sons(i).bb().col(0).array());
            bbmat.col(1).array() =
                bbmat.col(1).array().max(CT.sons(i).bb().col(1).array());
          }
        }
      } else {
        for (Index i = 0; i < CT.block_size(); ++i) {
          // determine minimum
          bbmat.col(0).array() =
              P.col(CT.indices()[i]).array().min(bbmat.col(0).array());
          // determine maximum
          bbmat.col(1).array() =
              P.col(CT.indices()[i]).array().max(bbmat.col(1).array());
        }
      }
    } else {
      // set everything to inf;
      bbmat.setOnes();
      bbmat.col(0) *= FMCA_INF;
      bbmat.col(1) *= -FMCA_INF;
    }
    bbmat.col(2) = bbmat.col(1) - bbmat.col(0);
    // fix potential flat bounding boxes
    for (Index i = 0; i < bbmat.rows(); ++i)
      if (bbmat(i, 1) - bbmat(i, 0) < 2e2 * FMCA_ZERO_TOLERANCE) {
        bbmat(i, 1) += 1e2 * FMCA_BBOX_THREASHOLD;
        bbmat(i, 0) -= 1e2 * FMCA_BBOX_THREASHOLD;
        bbmat(i, 2) = bbmat(i, 1) - bbmat(i, 0);
      }

    CT.node().bb_ = bbmat;
    return;
  }
  static Scalar geodesicDistance(const Vector &a, const Vector &b) {
    const Scalar dot = a.dot(b);
    const Scalar clamped_dot = std::min(1., std::max(-1., dot));
    return std::acos(clamped_dot);
  }
  /**
   *  \brief implements [K. Aftab, R. Hartley, J. Trumpf: Generalized Weiszfeld
   *                     Algorithms for Lq Optimization] for the special case
   *         of the unit sphere. Replacing the 1/d approximation by
   *         1/sin(d).
   *
   **/
  template <typename Derived>
  static Vector sphericalMedian(const Matrix &P,
                                const ClusterTreeBase<Derived> &node,
                                const Vector &c0, Index max_iter = 10,
                                const Scalar tol = FMCA_ZERO_TOLERANCE) {
    if (!node.block_size()) return c0;
    Vector c = c0;
    Vector c_old = c;
    Index iter = 0;
    do {
      c_old = c;
      c.setZero();
      for (Index i = 0; i < node.block_size(); ++i) {
        const Scalar d_i = geodesicDistance(c, P.col(node.indices()[i]));
        const Scalar w_i =
            1.0 / std::max(std::sin(d_i), 100 * FMCA_ZERO_TOLERANCE);
        c += w_i * P.col(node.indices()[i]);
      }
      c.normalize();
      ++iter;
    } while ((iter < max_iter) && ((c - c_old).norm() > tol));
    return c;
  }

  template <typename Derived>
  static void computeCR(const Matrix &P, ClusterTreeBase<Derived> &node) {
    for (Index i = 0; i < node.nSons(); ++i) computeCR(P, node.sons(i));
    node.node().c_ = Vector::Zero(P.rows());
    node.node().r_ = 0;
    if (!node.nSons())
      for (Index i = 0; i < node.block_size(); ++i)
        node.node().c_ += P.col(node.indices()[i]);
    else
      for (Index i = 0; i < node.nSons(); ++i)
        node.node().c_ += node.sons(i).node().c_;
    node.node().c_.normalize();
    node.node().c_ = sphericalMedian(P, node, node.node().c_, 10);
    for (Index i = 0; i < node.block_size(); ++i) {
      const Scalar dist =
          geodesicDistance(node.node().c_, P.col(node.indices()[i]));
      node.node().r_ = node.node().r_ < dist ? dist : node.node().r_;
    }
  }
};
}  // namespace internal

}  // namespace FMCA

#endif

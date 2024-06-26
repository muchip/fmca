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
#ifndef FMCA_CLUSTERING_ADMISSIBILITY_H_
#define FMCA_CLUSTERING_ADMISSIBILITY_H_

namespace FMCA {

enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };

template <typename Derived, typename otherDerived>
Scalar computeDistance(const ClusterTreeBase<Derived> &cluster1,
                       const ClusterTreeBase<otherDerived> &cluster2) {
  const Scalar radius1 = 0.5 * cluster1.bb().col(2).norm();
  const Scalar radius2 = 0.5 * cluster2.bb().col(2).norm();
  const Scalar dist = (0.5 * (cluster1.bb().col(0) - cluster2.bb().col(0) +
                              cluster1.bb().col(1) - cluster2.bb().col(1)))
                          .norm() -
                      radius1 - radius2;
  return dist > 0 ? dist : 0;
}

/**
 *  \brief classical admissibility condition based on the relative
 *         distance of the bounding boxes
 **/
struct CompareClusterBB {
  template <typename Derived, typename otherDerived>
  static Admissibility compare(const ClusterTreeBase<Derived> &cluster1,
                               const ClusterTreeBase<otherDerived> &cluster2,
                               Scalar eta) {
    const FMCA::Vector u =
        (cluster1.bb().col(0) - cluster2.bb().col(1)).cwiseMax(0);
    const FMCA::Vector v =
        (cluster2.bb().col(0) - cluster1.bb().col(1)).cwiseMax(0);
    const Scalar dist = sqrt(u.squaredNorm() + v.squaredNorm());
    const Scalar row_radius = 0.5 * cluster1.bb().col(2).norm();
    const Scalar col_radius = 0.5 * cluster2.bb().col(2).norm();
    const Scalar radius = row_radius > col_radius ? row_radius : col_radius;
    if (radius > eta * dist) {
      // check if either cluster is a leaf in that case,
      // compute the full matrix block
      if (!cluster1.nSons() || !cluster2.nSons())
        return Dense;
      else
        return Refine;
    } else
      return LowRank;
  }
};
/**
 *  \brief classical admissibility condition based on the relative
 *         distance of the enclosing balls
 **/
struct CompareCluster {
  template <typename Derived, typename otherDerived>
  static Admissibility compare(const ClusterTreeBase<Derived> &cluster1,
                               const ClusterTreeBase<otherDerived> &cluster2,
                               Scalar eta) {
    const Scalar dist = computeDistance(cluster1, cluster2);
    const Scalar row_radius = 0.5 * cluster1.bb().col(2).norm();
    const Scalar col_radius = 0.5 * cluster2.bb().col(2).norm();
    const Scalar radius = row_radius > col_radius ? row_radius : col_radius;

    if (radius > eta * dist) {
      // check if either cluster is a leaf in that case,
      // compute the full matrix block
      if (!cluster1.nSons() || !cluster2.nSons())
        return Dense;
      else
        return Refine;
    } else
      return LowRank;
  }
};
/**
 *  \brief classical admissibility condition based on the relative
 *         distance of the enclosing balls where the last component is
 *         periodic in [0,1)
 **/
struct CompareClusterTimePeriodic {
  template <typename Derived, typename otherDerived>
  static Admissibility compare(const ClusterTreeBase<Derived> &c1,
                               const ClusterTreeBase<otherDerived> &c2,
                               Scalar eta) {
    Admissibility retval;
    const Index n = c1.bb().rows();
    const Scalar sinc1 = std::sin(FMCA_PI * c1.bb()(n - 1, 2));
    const Scalar sinc2 = std::sin(FMCA_PI * c2.bb()(n - 1, 2));
    const Scalar sincc = std::sin(
        FMCA_PI * std::abs(0.5 * (c1.bb()(n - 1, 0) - c2.bb()(n - 1, 0) +
                                  c1.bb()(n - 1, 1) - c2.bb()(n - 1, 1))));
    const Scalar row_radius =
        0.5 *
        std::sqrt(c1.bb().col(2).head(n - 1).squaredNorm() + sinc1 * sinc1);
    const Scalar col_radius =
        0.5 *
        std::sqrt(c2.bb().col(2).head(n - 1).squaredNorm() + sinc2 * sinc2);

    const Scalar signeddist =
        std::sqrt(
            (0.5 * (c1.bb().col(0).head(n - 1) - c2.bb().col(0).head(n - 1) +
                    c1.bb().col(1).head(n - 1) - c2.bb().col(1).head(n - 1)))
                .squaredNorm() +
            sincc * sincc) -
        row_radius - col_radius;

    const Scalar dist = signeddist > 0 ? signeddist : 0;
    const Scalar radius = row_radius > col_radius ? row_radius : col_radius;

    if (radius > eta * dist) {
      // check if either cluster is a leaf in that case,
      // compute the full matrix block
      if (!c1.nSons() || !c2.nSons())
        return Dense;
      else
        return Refine;
    } else
      return LowRank;
  }
};
}  // namespace FMCA
#endif

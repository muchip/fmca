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
struct CompareClusterStrict {
  template <typename Derived, typename otherDerived>
  static Admissibility compare(const ClusterTreeBase<Derived> &cluster1,
                               const ClusterTreeBase<otherDerived> &cluster2,
                               Scalar eta) {
    const bool A =
        (cluster1.bb().col(0).array() <= cluster2.bb().col(0).array()).all() &&
        (cluster2.bb().col(1).array() <= cluster1.bb().col(1).array()).all();
    const bool B =
        (cluster2.bb().col(0).array() <= cluster1.bb().col(0).array()).all() &&
        (cluster1.bb().col(1).array() <= cluster2.bb().col(1).array()).all();
    if (A || B) {
      // check if either cluster is a leaf in that case,
      // compute the full matrix block
      if (!cluster1.nSons() || !cluster2.nSons())
        return Dense;
      else
        return Refine;
    } else
      return LowRank;
  }
  static Scalar geodesicDistance(const Vector &a, const Vector &b) {
    const Scalar dot = a.dot(b);
    const Scalar clamped_dot = std::min(1., std::max(-1., dot));
    return std::acos(clamped_dot);
  }
};

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
 *  \brief geodesic admissibility condition based on the relative
 *         distance of enclosing balls on the sphere
 **/
struct CompareSphericalCluster {
  template <typename Derived, typename otherDerived>
  static Admissibility compare(const ClusterTreeBase<Derived> &cluster1,
                               const ClusterTreeBase<otherDerived> &cluster2,
                               Scalar eta) {
    // for now, we use an ugly typecast here
    Scalar dist = geodesicDistance(cluster1.node().c_, cluster2.node().c_);
    const Scalar row_radius = cluster1.node().r_;
    const Scalar col_radius = cluster2.node().r_;
    dist = dist - row_radius - col_radius;
    dist = dist > 0 ? dist : 0;
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
  static Scalar geodesicDistance(const Vector &a, const Vector &b) {
    const Scalar dot = a.dot(b);
    const Scalar clamped_dot = std::min(1., std::max(-1., dot));
    return std::acos(clamped_dot);
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

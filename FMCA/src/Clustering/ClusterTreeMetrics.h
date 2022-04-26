// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_CLUSTERING_CLUSTERTREEMETRICS_H_
#define FMCA_CLUSTERING_CLUSTERTREEMETRICS_H_

namespace FMCA {

enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };

template <typename Derived>
typename Derived::eigenMatrix::Scalar computeDistance(
    const ClusterTreeBase<Derived> &TR, const ClusterTreeBase<Derived> &TC) {
  typedef typename Derived::eigenMatrix::Scalar value_type;

  const value_type row_radius = 0.5 * TR.bb().col(2).norm();
  const value_type col_radius = 0.5 * TC.bb().col(2).norm();
  const value_type dist =
      0.5 * (TR.bb().col(0) - TC.bb().col(0) + TR.bb().col(1) - TC.bb().col(1))
                .norm() -
      row_radius - col_radius;
  return dist > 0 ? dist : 0;
}

template <typename Derived>
Admissibility compareCluster(const ClusterTreeBase<Derived> &cluster1,
                             const ClusterTreeBase<Derived> &cluster2,
                             typename Derived::eigenMatrix::Scalar eta) {
  typedef typename Derived::eigenMatrix::Scalar value_type;
  Admissibility retval;
  const value_type dist = computeDistance(cluster1, cluster2);
  const value_type row_radius = 0.5 * cluster1.bb().col(2).norm();
  const value_type col_radius = 0.5 * cluster2.bb().col(2).norm();
  const value_type radius = row_radius > col_radius ? row_radius : col_radius;

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

template <typename Derived, typename Derived2>
typename Derived2::Scalar clusterSeparationRadius(
    typename Derived2::Scalar separation_radius,
    const ClusterTreeBase<Derived> &cluster1,
    const ClusterTreeBase<Derived> &cluster2,
    const Eigen::MatrixBase<Derived2> &P) {
  typename Derived2::Scalar retval = separation_radius;
  typename Derived2::Scalar rad = 0;
  typename Derived2::Scalar dist = computeDistance(cluster1, cluster2);

  if (separation_radius > 0.5 * dist) {
    if (cluster2.nSons()) {
      rad = 0.5 * (dist + cluster2.bb().col(2).norm());
      separation_radius = separation_radius < rad ? separation_radius : rad;
      for (auto i = 0; i < cluster2.nSons(); ++i) {
        rad = clusterSeparationRadius(separation_radius, cluster1,
                                      cluster2.sons(i), P);
        retval = retval < rad ? retval : rad;
      }
    } else {
      if (cluster1.block_id() != cluster2.block_id())
        for (auto j = 0; j < cluster1.indices().size(); ++j)
          for (auto i = 0; i < cluster2.indices().size(); ++i) {
            rad = 0.5 *
                  (P.col(cluster2.indices()[i]) - P.col(cluster1.indices()[j]))
                      .norm();
            retval = retval < rad ? retval : rad;
          }
    }
  }
  return retval;
}

template <typename Derived, typename Derived2>
void clusterFillDistance(
    Eigen::Matrix<typename Derived2::Scalar, Eigen::Dynamic, 1> &fill_distance,
    const ClusterTreeBase<Derived> &cluster1,
    const ClusterTreeBase<Derived> &cluster2,
    const Eigen::MatrixBase<Derived2> &P) {
  typename Derived2::Scalar max_dist = 0;
  typename Derived2::Scalar dist = computeDistance(cluster1, cluster2);
  // check cluster if there is the chance of improving the distance of a given
  // point
  if (fill_distance.maxCoeff() > dist) {
    if (cluster2.nSons()) {
      for (auto i = 0; i < cluster2.nSons(); ++i)
        clusterFillDistance(fill_distance, cluster1, cluster2.sons(i), P);
    } else {
      if (cluster1.block_id() != cluster2.block_id())
        for (auto j = 0; j < cluster1.indices().size(); ++j)
          for (auto i = 0; i < cluster2.indices().size(); ++i) {
            max_dist =
                (P.col(cluster2.indices()[i]) - P.col(cluster1.indices()[j]))
                    .norm();
            fill_distance(j) =
                fill_distance(j) > max_dist ? max_dist : fill_distance(j);
          }
    }
  }
  return;
}

template <typename Derived, typename Derived2>
typename Derived2::Scalar separationRadius(
    const ClusterTreeBase<Derived> &CT, const Eigen::MatrixBase<Derived2> &P) {
  typedef typename Derived2::Scalar value_type;
  value_type separation_radius = value_type(1.) / value_type(0.);
  for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    if (!it->nSons()) {
      value_type rad = 0;
      for (auto j = 0; j < it->indices().size(); ++j)
        for (auto i = j + 1; i < it->indices().size(); ++i) {
          rad =
              0.5 * (P.col(it->indices()[i]) - P.col(it->indices()[j])).norm();
          separation_radius = separation_radius < rad ? separation_radius : rad;
        }
      rad = clusterSeparationRadius(separation_radius, *it, CT, P);
      separation_radius = separation_radius < rad ? separation_radius : rad;
    }
  }
  return separation_radius;
}

template <typename Derived, typename Derived2>
typename Derived2::Scalar fillDistance(const ClusterTreeBase<Derived> &CT,
                                       const Eigen::MatrixBase<Derived2> &P) {
  typedef typename Derived2::Scalar value_type;
  value_type fill_distance = value_type(0.);
  Eigen::Matrix<value_type, Eigen::Dynamic, 1> min_distance;
  value_type dist = 0;
  for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    if (!it->nSons()) {
      min_distance.resize(it->indices().size());
      min_distance.setOnes();
      min_distance /= value_type(0.);
      // determine candidate distances within the cluster
      for (auto j = 0; j < it->indices().size(); ++j)
        for (auto i = 0; i < it->indices().size(); ++i) {
          if (i != j) {
            dist = (P.col(it->indices()[i]) - P.col(it->indices()[j])).norm();
            min_distance(j) = min_distance(j) > dist ? dist : min_distance(j);
          }
        }
      clusterFillDistance(min_distance, *it, CT, P);
      const value_type cluster_max_dist = min_distance.maxCoeff();
      fill_distance =
          fill_distance < cluster_max_dist ? cluster_max_dist : fill_distance;
    }
  }
  return fill_distance;
}

}  // namespace FMCA
#endif

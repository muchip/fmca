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
#ifndef FMCA_CLUSTERING_CLUSTERTREEMETRICS_H_
#define FMCA_CLUSTERING_CLUSTERTREEMETRICS_H_

#include <memory>

#include "../util/Tictoc.h"

namespace FMCA {

enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };

template <typename Derived, typename Derived2>
Scalar pointDistance(const ClusterTreeBase<Derived> &TR,
                     const Eigen::MatrixBase<Derived2> &pt) {
  const Scalar radius = 0.5 * TR.bb().col(2).norm();
  const Scalar dist = (0.5 * (TR.bb().col(0) + TR.bb().col(1)) - pt) - radius;
  return dist > 0 ? dist : 0;
}

template <typename Derived, typename Derived2>
bool inBoundingBox(const ClusterTreeBase<Derived> &TR,
                   const Eigen::MatrixBase<Derived2> &pt) {
  return ((pt - TR.bb().col(0)).array() >= 0).all() *
         ((pt - TR.bb().col(1)).array() <= 0).all();
}

template <typename Derived>
Scalar computeDistance(const ClusterTreeBase<Derived> &TR,
                       const ClusterTreeBase<Derived> &TC) {
  const Scalar row_radius = 0.5 * TR.bb().col(2).norm();
  const Scalar col_radius = 0.5 * TC.bb().col(2).norm();
  const Scalar dist =
      0.5 * (TR.bb().col(0) - TC.bb().col(0) + TR.bb().col(1) - TC.bb().col(1))
                .norm() -
      row_radius - col_radius;
  return dist > 0 ? dist : 0;
}

template <typename Derived>
Admissibility compareCluster(const ClusterTreeBase<Derived> &cluster1,
                             const ClusterTreeBase<Derived> &cluster2,
                             Scalar eta) {
  Admissibility retval;
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

template <typename Derived>
void updateClusterMinDistance(Vector &min_dist, Scalar max_min_dist,
                              const ClusterTreeBase<Derived> &c1,
                              const ClusterTreeBase<Derived> &c2,
                              const Matrix &P) {
  Scalar dist = computeDistance(c1, c2);
  if (max_min_dist >= dist) {
    if (c2.nSons()) {
      dist += c2.bb().col(2).norm();
      max_min_dist = max_min_dist < dist ? max_min_dist : dist;
      for (Index i = 0; i < c2.nSons(); ++i)
        if (c2.sons(i).indices().size())
          updateClusterMinDistance(min_dist, max_min_dist, c1, c2.sons(i), P);
    } else {
      if (c1.block_id() < c2.block_id())
        for (Index j = 0; j < c1.indices().size(); ++j)
          for (Index i = 0; i < c2.indices().size(); ++i) {
            dist = (P.col(c2.indices()[i]) - P.col(c1.indices()[j])).norm();
            min_dist(c2.indices()[i]) = min_dist(c2.indices()[i]) > dist
                                            ? dist
                                            : min_dist(c2.indices()[i]);
            min_dist(c1.indices()[j]) = min_dist(c1.indices()[j]) > dist
                                            ? dist
                                            : min_dist(c1.indices()[j]);
          }
      // determin max_min_distance within cluster
      max_min_dist = 0;
      for (Index j = 0; j < c1.indices().size(); ++j)
        max_min_dist = max_min_dist < min_dist(c1.indices()[j])
                           ? min_dist(c1.indices()[j])
                           : max_min_dist;
    }
  }
  return;
}

template <typename Derived>
Vector minDistanceVector(const ClusterTreeBase<Derived> &CT, const Matrix &P) {
  Vector min_distance(P.cols());
  Scalar max_min_distance = 0;
  min_distance.setOnes();
  min_distance /= Scalar(0.);
  Scalar dist = 0;
  // compute min_distance at the leafs
  for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    if (!it->nSons() && it->indices().size()) {
      // determine min_distances within the cluster
      for (Index j = 0; j < it->indices().size(); ++j)
        for (Index i = 0; i < j; ++i) {
          dist = (P.col(it->indices()[i]) - P.col(it->indices()[j])).norm();
          min_distance(it->indices()[j]) = min_distance(it->indices()[j]) > dist
                                               ? dist
                                               : min_distance(it->indices()[j]);
          min_distance(it->indices()[i]) = min_distance(it->indices()[i]) > dist
                                               ? dist
                                               : min_distance(it->indices()[i]);
        }
      // determin max_min_distance within cluster
      max_min_distance = 0;
      for (Index j = 0; j < it->indices().size(); ++j)
        max_min_distance = max_min_distance < min_distance(it->indices()[j])
                               ? min_distance(it->indices()[j])
                               : max_min_distance;
      updateClusterMinDistance(min_distance, max_min_distance, *it, CT, P);
    }
  }
  return min_distance;
}

template <typename Derived>
void clusterTreeStatistics(const ClusterTreeBase<Derived> &CT,
                           const Matrix &P) {
  std::vector<Scalar> disc_vec;
  Index N = CT.indices().size();
  Index n_cluster = std::distance(CT.cbegin(), CT.cend());
  Scalar vol = CT.bb().col(2).prod();
  Scalar max_disc = 0;
  Scalar min_disc = FMCA_INF;
  Scalar mean_disc = 0;
  Vector min_dist = minDistanceVector(CT, P);
  Scalar min_min_dist = min_dist.minCoeff();
  Scalar max_min_dist = min_dist.maxCoeff();
  if (n_cluster > 1) {
    for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
      const auto &node = *it;
      if (!node.is_root() && node.indices().size()) {
        Scalar discrepancy = std::abs(Scalar(node.indices().size()) / N -
                                      node.bb().col(2).prod() / vol);
        disc_vec.push_back(discrepancy);
        max_disc = max_disc < disc_vec.back() ? disc_vec.back() : max_disc;
        min_disc = min_disc > disc_vec.back() ? disc_vec.back() : min_disc;
        mean_disc += disc_vec.back();
      }
    }
    mean_disc /= disc_vec.size();
  } else {
    min_disc = FMCA_ZERO_TOLERANCE;
    max_disc = FMCA_ZERO_TOLERANCE;
    mean_disc = FMCA_ZERO_TOLERANCE;
  }

  std::cout << "------------------- Cluster tree metrics -------------------"
            << std::endl;
  std::cout << "dimension:                    " << P.rows() << std::endl;
  std::cout << "number of points:             " << N << std::endl;
  std::cout << "cluster splitting method:     "
            << FMCA::internal::traits<Derived>::Splitter::splitterName()
            << std::endl;
  std::cout << "bounding box diameter:        " << CT.bb().col(2).norm()
            << std::endl;
  std::cout << "number of clusters:           " << n_cluster << std::endl;
  std::cout << "fill distance:                " << max_min_dist << std::endl;
  std::cout << "separation radius:            " << 0.5 * min_min_dist
            << std::endl;
  std::cout << std::scientific << std::setprecision(2)
            << "min cluster discrepancy:      " << min_disc << std::endl
            << "max cluster discrepancy:      " << max_disc << std::endl
            << "mean cluster discrepancy:     " << mean_disc << std::endl;
  {
    std::cout << "pt. mindist distribution:     " << std::endl;
    FMCA::Index intervals = 15;
    FMCA::Scalar h = (max_min_dist - min_min_dist) / intervals;
    FMCA::Vector values(intervals);
    values.setZero();
    for (auto i = 0; i < min_dist.size(); ++i) {
      auto j = 0;
      for (; j < intervals; ++j)
        if ((min_dist(i) >= h * j + min_min_dist) &&
            (min_dist(i) < h * (j + 1) + min_min_dist)) {
          ++values(j);
          break;
        }
    }
    FMCA::Scalar bar_factor = 40 * min_dist.size() / values.maxCoeff();
    bar_factor = bar_factor < FMCA_INF ? bar_factor : 0;
    for (auto i = 0; i < intervals; ++i) {
      std::cout << std::scientific << std::setprecision(2) << std::setw(9)
                << h * (i + 0.5) + min_min_dist << "|";
      std::cout << std::string(
                       std::ceil(bar_factor * values(i) / min_dist.size()), '*')
                << std::endl;
    }
  }

  {
    std::cout << "discrepancy distribution:     " << std::endl;
    FMCA::Index intervals = 15;
    FMCA::Scalar h = log(max_disc / min_disc) / intervals;
    FMCA::Vector values(intervals);
    values.setZero();
    for (auto &&it : disc_vec) {
      auto j = 0;
      for (; j < intervals; ++j)
        if ((log(it) >= h * j + log(min_disc)) &&
            (log(it) < h * (j + 1) + log(min_disc))) {
          ++values(j);
          break;
        }
    }
    FMCA::Scalar bar_factor = 40 * disc_vec.size() / values.maxCoeff();
    bar_factor = bar_factor < FMCA_INF ? bar_factor : 0;
    for (auto i = 0; i < intervals; ++i) {
      std::cout << std::scientific << std::setprecision(2) << std::setw(9)
                << exp(h * (i + 0.5)) * min_disc << "|";
      std::cout << std::string(
                       std::ceil(bar_factor * values(i) / disc_vec.size()), '*')
                << std::endl;
    }
  }
  std::cout << std::string(60, '-') << std::endl;
  return;
}

}  // namespace FMCA
#endif

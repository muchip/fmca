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
Scalar clusterSeparationRadius(Scalar separation_radius,
                               const ClusterTreeBase<Derived> &cluster1,
                               const ClusterTreeBase<Derived> &cluster2,
                               const Matrix &P) {
  Scalar retval = separation_radius;
  Scalar rad = 0;
  Scalar dist = computeDistance(cluster1, cluster2);
  if (separation_radius >= 0.5 * dist) {
    if (cluster2.nSons()) {
      rad = 0.5 * (dist + cluster2.bb().col(2).norm());
      separation_radius = separation_radius < rad ? separation_radius : rad;
      for (auto i = 0; i < cluster2.nSons(); ++i) {
        if (cluster2.sons(i).indices().size())
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

template <typename Derived>
void clusterFillDistance(Vector &min_distance,
                         const ClusterTreeBase<Derived> &cluster1,
                         const ClusterTreeBase<Derived> &cluster2,
                         const Matrix &P) {
  Scalar dist = computeDistance(cluster1, cluster2);
  // check cluster if there is the chance of improving the distance of a
  // given point
  if (min_distance.maxCoeff() >= dist) {
    if (cluster2.nSons()) {
      for (auto i = 0; i < cluster2.nSons(); ++i)
        if (cluster2.sons(i).indices().size())
          clusterFillDistance(min_distance, cluster1, cluster2.sons(i), P);
    } else {
      if (cluster1.block_id() != cluster2.block_id())
        for (auto j = 0; j < cluster1.indices().size(); ++j)
          for (auto i = 0; i < cluster2.indices().size(); ++i) {
            dist = (P.col(cluster2.indices()[i]) - P.col(cluster1.indices()[j]))
                       .norm();
            min_distance(j) = min_distance(j) > dist ? dist : min_distance(j);
          }
    }
  }
  return;
}

template <typename Derived>
Scalar separationRadius(const ClusterTreeBase<Derived> &CT, const Matrix &P) {
  Scalar separation_radius = Scalar(1. / 0.);
  for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    if (!it->nSons() && it->indices().size()) {
      Scalar rad = 0;
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

template <typename Derived>
Scalar fillDistance(const ClusterTreeBase<Derived> &CT, const Matrix &P) {
  Scalar fill_distance = Scalar(0.);
  Vector min_distance;
  Scalar dist = 0;
  for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    if (!it->nSons() && it->indices().size()) {
      min_distance.resize(it->indices().size());
      min_distance.setOnes();
      min_distance /= Scalar(0.);
      // determine candidate distances within the cluster
      for (auto j = 0; j < it->indices().size(); ++j)
        for (auto i = 0; i < j; ++i) {
          dist = (P.col(it->indices()[i]) - P.col(it->indices()[j])).norm();
          min_distance(j) = min_distance(j) > dist ? dist : min_distance(j);
          min_distance(i) = min_distance(i) > dist ? dist : min_distance(i);
        }
      clusterFillDistance(min_distance, *it, CT, P);
      const Scalar cluster_max_dist = min_distance.maxCoeff();
      fill_distance =
          fill_distance < cluster_max_dist ? cluster_max_dist : fill_distance;
    }
  }
  return fill_distance;
}

template <typename Derived>
void clusterTreeStatistics(const ClusterTreeBase<Derived> &CT,
                           const Matrix &P) {
  std::vector<Scalar> disc_vec;
  Index N = CT.indices().size();
  Index n_cluster = std::distance(CT.cbegin(), CT.cend());
  Scalar vol = CT.bb().col(2).prod();
  Scalar max_disc = 0;
  Scalar min_disc = 1. / 0.;
  Scalar mean_disc = 0;

  for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    const auto &node = *it;
    if (node.indices().size() > 1) {
      Scalar discrepancy = std::abs(Scalar(node.indices().size()) / N -
                                    node.bb().col(2).prod() / vol);
      disc_vec.push_back(discrepancy);
      max_disc = max_disc < disc_vec.back() ? disc_vec.back() : max_disc;
      min_disc = min_disc > disc_vec.back() ? disc_vec.back() : min_disc;
      mean_disc += disc_vec.back();
    }
  }
  mean_disc /= disc_vec.size();
  std::cout << "number of clusters:         " << n_cluster << std::endl;
  std::cout << "number of actual clusters:  " << disc_vec.size() << std::endl;
  std::cout << "fill distance:              " << FMCA::fillDistance(CT, P)
            << std::endl;
  std::cout << "sep radius:                 " << FMCA::separationRadius(CT, P)
            << std::endl;
  std::cout << std::scientific << std::setprecision(2)
            << "min cluster discrepancy:    " << min_disc << std::endl
            << "max cluster discrepancy:    " << max_disc << std::endl
            << "mean cluster discrepancy:   " << mean_disc << std::endl;
  FMCA::Scalar range = exp(log(mean_disc) - 2 * log(max_disc / mean_disc));
  min_disc = min_disc < range ? range : min_disc;
  std::cout << std::scientific << std::setprecision(2)
            << "cluster discrepancy distribution in [" << min_disc << ","
            << max_disc << ")" << std::endl;
  FMCA::Index intervals = 20;
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
  FMCA::Scalar bar_factor = 60 * disc_vec.size() / values.maxCoeff();
  bar_factor = bar_factor < Scalar(1. / 0.) ? bar_factor : 0;
  for (auto i = 0; i < intervals; ++i) {
    std::cout << std::scientific << std::setprecision(2) << std::setw(8)
              << exp(h * (i + 0.5)) * min_disc << "|";
    std::cout << std::string(
                     std::round(bar_factor * values(i) / disc_vec.size()), '=')
              << std::endl;
  }
}

} // namespace FMCA
#endif

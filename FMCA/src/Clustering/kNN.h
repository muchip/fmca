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
#ifndef FMCA_CLUSTERING_KNN_H_
#define FMCA_CLUSTERING_KNN_H_

#include <memory>

#include "../util/KMinList.h"

namespace FMCA {

template <typename Derived>
void updateClusterKMinDistance(std::vector<KMinList> &min_dist,
                               Scalar max_min_dist,
                               const ClusterTreeBase<Derived> &c1,
                               const ClusterTreeBase<Derived> &c2,
                               const Matrix &P) {
  Scalar dist = computeDistance(c1, c2);
  if (max_min_dist >= dist) {
    if (c2.nSons()) {
      dist += c2.bb().col(2).norm();
      // max_min_dist = max_min_dist < dist ? max_min_dist : dist;
      for (Index i = 0; i < c2.nSons(); ++i)
        if (c2.sons(i).indices().size())
          updateClusterKMinDistance(min_dist, max_min_dist, c1, c2.sons(i), P);
    } else {
      if (c1.block_id() != c2.block_id())
        for (Index j = 0; j < c1.indices().size(); ++j)
          for (Index i = 0; i < c2.indices().size(); ++i) {
            dist = (P.col(c2.indices()[i]) - P.col(c1.indices()[j])).norm();
            min_dist[c1.indices()[j]].insert(
                std::make_pair(c2.indices()[i], dist));
          }
      // determine max_min_distance within cluster
      max_min_dist = 0;
      for (Index j = 0; j < c1.indices().size(); ++j) {
        const Scalar dist = min_dist[c1.indices()[j]].max_min();
        max_min_dist = max_min_dist < dist ? dist : max_min_dist;
      }
    }
  }
  return;
}

template <typename Derived>
iMatrix kNN(const ClusterTreeBase<Derived> &CT, const Matrix &P,
            const Index k = 1) {
  assert(k < P.cols() && "too few data points for kmin");
  iMatrix kmin_distance(P.cols(), k);
  std::vector<KMinList> qvec(P.cols());
#pragma omp parallel for
  for (Index i = 0; i < qvec.size(); ++i) {
    qvec[i] = KMinList(k);
    for (Index j = 1; j <= k; ++j) {
      const Index idx = (i + j) % P.cols();
      assert(idx != i && "should never happen");
      const Scalar dist = (P.col(i) - P.col(idx)).norm();
      qvec[i].insert(std::make_pair(idx, dist));
    }
  }
  std::vector<const Derived *> plist;
  plist.reserve(P.cols());
  for (auto it = CT.cbegin(); it != CT.cend(); ++it)
    if (!it->nSons() && it->indices().size())
      plist.push_back(std::addressof(*it));
      // compute min_distance at the leafs
#pragma omp parallel for
  for (auto it = plist.begin(); it != plist.end(); ++it) {
    // for (auto it = CT.cbegin(); it != CT.cend(); ++it) {
    // if (!it->nSons() && it->indices().size()) {
    const std::vector<Index> &idcs = (*it)->indices();
    for (Index j = 0; j < idcs.size(); ++j)
      for (Index i = 0; i < j; ++i) {
        const Scalar dist = (P.col(idcs[i]) - P.col(idcs[j])).norm();
        qvec[idcs[i]].insert(std::make_pair(idcs[j], dist));
        qvec[idcs[j]].insert(std::make_pair(idcs[i], dist));
      }
    // determine max_min_distance within cluster
    Scalar max_min_distance = 0;
    for (Index j = 0; j < idcs.size(); ++j) {
      const Scalar dist = qvec[idcs[j]].max_min();
      max_min_distance = max_min_distance < dist ? dist : max_min_distance;
    }
    assert(max_min_distance > 0 && "min_dist should be positive");
    updateClusterKMinDistance(qvec, max_min_distance, **it, CT, P);
    //}
  }
#pragma omp parallel for
  for (Index i = 0; i < qvec.size(); ++i) {
    Index j = 0;
    for (const auto &it : qvec[i].list()) kmin_distance(i, j++) = it.first;
  }
  return kmin_distance;
}

}  // namespace FMCA
#endif

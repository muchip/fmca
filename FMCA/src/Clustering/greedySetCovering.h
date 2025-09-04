// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_CLUSTERING_GREEDYSETCOVERING_H_
#define FMCA_CLUSTERING_GREEDYSETCOVERING_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief uses the cluster tree ct to efficiently determine a
 *         set covering of a given radius
 **/
template <typename Derived>
std::vector<Index> greedySetCovering(const ClusterTreeBase<Derived> &ct,
                                     const Matrix &P, const Scalar r) {
  std::vector<Index> retval;
  std::vector<bool> is_covered(P.cols(), false);
  std::vector<Index> n_uncovered(P.cols());
  std::vector<std::vector<Index>> rballs(P.cols());
  std::vector<std::vector<Index>> index_covers(P.cols());
  Index num_covered = 0;

#pragma omp parallel for
  for (Index i = 0; i < rballs.size(); ++i) {
    rballs[i] = epsNN(ct, P, P.col(i), 0.5 * r);
    n_uncovered[i] = rballs[i].size();
    for (const auto &it : rballs[i])
#pragma omp critical
      index_covers[it].push_back(i);
  }

  while (num_covered != P.cols()) {
    Index max_size = 0;
    Index max_index = -1;
    // determine largest ball
    for (Index i = 0; i < rballs.size(); ++i) {
      max_index = max_size < n_uncovered[i] ? i : max_index;
      max_size = max_size < n_uncovered[i] ? n_uncovered[i] : max_size;
    }
    // store selected ball
    num_covered += max_size;
    retval.push_back(max_index);
    for (const auto &it : rballs[max_index]) {
      // reduce uncovered size of affected balls
      if (!is_covered[it]) {
        for (const auto &it2 : index_covers[it]) n_uncovered[it2] -= 1;
        // mask covered indices
        is_covered[it] = true;
      }
    }
  }
  return retval;
}

}  // namespace FMCA
#endif

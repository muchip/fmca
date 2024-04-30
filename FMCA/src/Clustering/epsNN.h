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
#ifndef FMCA_CLUSTERING_EPSNN_H_
#define FMCA_CLUSTERING_EPSNN_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief uses the cluster tree ct to efficiently determined the indices
 *         of all points in P that are less than eps away from pt.
 **/
template <typename Derived>
std::vector<Index> epsNN(const ClusterTreeBase<Derived> &ct, const Matrix &P,
                         const Vector &pt, const Scalar eps) {
  std::vector<Index> retval;
  std::vector<const Derived *> queue;
  queue.push_back(std::addressof(ct.derived()));
  while (queue.size()) {
    const Derived &nd = *(queue.back());
    queue.pop_back();
    if (!nd.nSons()) {
      // at all indices to the result
      for (Index i = 0; i < nd.block_size(); ++i)
        if ((pt - P.col(nd.indices()[i])).norm() < eps)
          retval.push_back(nd.indices()[i]);
    } else {
      for (Index i = 0; i < nd.nSons(); ++i) {
        const Vector bpt = (nd.sons(i).bb().col(0).cwiseMax(pt))
                               .cwiseMin(nd.sons(i).bb().col(1));
        if ((pt - bpt).norm() < eps)
          queue.push_back(std::addressof(nd.sons(i)));
      }
    }
  }
  return retval;
}

}  // namespace FMCA
#endif

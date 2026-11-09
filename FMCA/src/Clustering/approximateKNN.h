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
#ifndef FMCA_CLUSTERING_APPROXIMATEKNN_H_
#define FMCA_CLUSTERING_APPROXIMATEKNN_H_

#include "../util/KMinList.h"

namespace FMCA {

std::vector<Triplet> approximateSymKNN(const Matrix &P, const Index k = 1,
                                       const Index leaf_size = 100,
                                       const Index n_trees = 10) {
  assert(k < P.cols() && "too few data points for kmin");
  std::vector<Triplet> retval;
  std::vector<KMinList> qvec(P.cols());
#pragma omp parallel for
  for (Index i = 0; i < qvec.size(); ++i) qvec[i] = KMinList(k);
  std::vector<std::vector<KMinList>> thread_qvecs;

#pragma omp parallel
  {
    std::vector<KMinList> local_qvec(P.cols());
    for (Index i = 0; i < P.cols(); ++i) local_qvec[i] = KMinList(k);
#pragma omp for schedule(dynamic)
    for (Index t = 0; t < n_trees; ++t) {
      RandomProjectionTree ct(P, leaf_size);
      for (const auto &it : ct) {
        if (!it.nSons()) {
          for (Index j = 0; j < it.block_size(); ++j) {
            for (Index i = 0; i < j; ++i) {
              const Scalar d =
                  (P.col(it.indices()[i]) - P.col(it.indices()[j])).norm();
              local_qvec[it.indices()[i]].insert(
                  std::make_pair(it.indices()[j], d));
              local_qvec[it.indices()[j]].insert(
                  std::make_pair(it.indices()[i], d));
            }
          }
        }
      }
    }
#pragma omp critical
    thread_qvecs.push_back(std::move(local_qvec));
  }

#pragma omp parallel for
  for (Index i = 0; i < P.cols(); ++i) {
    for (const auto &local_qvec : thread_qvecs) {
      for (const auto &it : local_qvec[i].list()) {
        qvec[i].insert(it);
      }
    }
  }

  retval.reserve(2 * P.cols() * k);
  for (Index i = 0; i < qvec.size(); ++i)
    for (const auto &it : qvec[i].list())
      if (i > it.first) {
        retval.push_back(Triplet(i, it.first, it.second));
        retval.push_back(Triplet(it.first, i, it.second));
      }
  retval.shrink_to_fit();
  return retval;
}

}  // namespace FMCA
#endif

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
#ifndef FMCA_CLUSTERING_UNIFORMSUBSAMPLE_H_
#define FMCA_CLUSTERING_UNIFORMSUBSAMPLE_H_

namespace FMCA {

std::vector<Index> uniformSubsample(const std::vector<Index> &ids,
                                    const Matrix &P, const Index level) {
  assert(P.rows() < 4 && "this function only makes sense for d < 4");
  const Vector min = P.rowwise().minCoeff().array() - 10 * FMCA_ZERO_TOLERANCE;
  const Vector max = P.rowwise().maxCoeff().array() + 10 * FMCA_ZERO_TOLERANCE;
  const Vector dist = max - min;
  const Index n = 1 << level;
  const size_t np = std::pow(n, P.rows());
  std::vector<std::pair<Index, Scalar>> boxes(np);
  std::vector<Index> cids;
  std::vector<Index> indices = ids;
  indices.reserve(np + ids.size());
  // determine the indices that are not present in ids (cost O(N))
  {
    std::vector<Index> allids(P.cols());
    std::iota(allids.begin(), allids.end(), 0);
    std::set_difference(allids.begin(), allids.end(), ids.begin(), ids.end(),
                        std::inserter(cids, cids.begin()));
  }
  for (Index i = 0; i < boxes.size(); ++i) {
    boxes[i].first = -1;
    boxes[i].second = FMCA_INF;
  }

  // now fill the array
  for (const auto &it : cids) {
    // compute linear index
    Vector mp(P.rows());
    Index ind = 0;
    Index coord = Index(n * (P(0, it) - min(0)) / dist(0));
    mp(0) = Scalar(coord + 0.5) / n;
    ind = coord;
    for (Index j = 1; j < P.rows(); ++j) {
      coord = Index(n * (P(j, it) - min(j)) / dist(j));
      ind = n * ind + coord;
      mp(j) = Scalar(coord + 0.5) / n;
    }
    Scalar mp_dist =
        (((P.col(it) - min).array() / dist.array()).matrix() - mp).norm();
    if (boxes[ind].second > mp_dist) {
      boxes[ind].first = it;
      boxes[ind].second = mp_dist;
    }
  }
  for (const auto &it : boxes)
    if (it.first != -1) indices.push_back(it.first);
  indices.shrink_to_fit();
  std::sort(indices.begin(), indices.end());
  return indices;
}
}  // namespace FMCA
#endif

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
  const Vector min = P.rowwise().minCoeff().array();
  const Vector max = P.rowwise().maxCoeff().array();
  const Vector dist = max - min;
  const Index n = 1 << level;
  const size_t np = std::round(std::pow(n, P.rows()));
  std::vector<std::pair<std::ptrdiff_t, Scalar>> boxes(np);
  std::vector<Index> cids;
  std::vector<Index> indices = ids;
  indices.reserve(np + ids.size());
  // determine the indices that are not present in ids (cost O(N))
  {
    std::vector<Index> allids(P.cols());
    std::vector<Index> sids = ids;
    std::iota(allids.begin(), allids.end(), 0);
    std::sort(sids.begin(), sids.end());
    std::set_difference(allids.begin(), allids.end(), sids.begin(), sids.end(),
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
    coord = std::max<Index>(0, std::min<Index>(n - 1, coord));
    mp(0) = Scalar(coord + 0.5) / n;
    ind = coord;
    for (Index j = 1; j < P.rows(); ++j) {
      coord = Index(n * (P(j, it) - min(j)) / dist(j));
      coord = std::max<Index>(0, std::min<Index>(n - 1, coord));
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
  return indices;
}
}  // namespace FMCA
#endif

#if 0
  for (Index b = 0; b < boxes.size(); ++b) {
    auto const &box = boxes[b];
    if (box.first == -1) continue;

    const Index k = box.first;
    Vector x_norm = (P.col(k).array() - min.array()) / dist.array();

    // decode box index -> coordinates
    Index tmp = b;
    Vector c(P.rows());
    for (Index j = P.rows() - 1; j >= 0; --j) {
      c(j) = tmp % n;
      tmp /= n;
      if (j == 0) break;
    }
    Vector mp = (c.array() + 0.5) / Scalar(n);

    std::cout << "box " << b << " k " << k << " x_norm " << x_norm.transpose()
              << " mp " << mp.transpose() << " dist " << box.second << "\n";
  }
#endif

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

std::vector<Index> uniformSubsample(const std::vector<Index> &fxd_ids,
                                    const Matrix &P, const Index level) {
  assert(P.rows() < 4 && "this function only makes sense for d < 4");
  const Vector min = P.rowwise().minCoeff().array() - 10 * FMCA_ZERO_TOLERANCE;
  const Vector max = P.rowwise().maxCoeff().array() + 10 * FMCA_ZERO_TOLERANCE;
  const Index n = 1 << level;
  const size_t np = std::pow(n, P.rows());
  std::vector<bool> boxes(np);
  std::vector<Index> indices;
  indices.reserve(P.cols());
  // first insert fixed indices
  for (Index i = 0; i < fxd_ids.size(); ++i) {
    // compute linear index
    size_t ind = Index(n * (P(0, fxd_ids[i]) - min(0)) / (max(0) - min(0)));
    for (Index j = 1; j < P.rows(); ++j)
      ind =
          n * ind + Index(n * (P(j, fxd_ids[i]) - min(j)) / (max(j) - min(j)));
    boxes[ind] = true;
    indices.push_back(fxd_ids[i]);
  }
  // now fill the array
  for (Index i = 0; i < P.cols(); ++i) {
    // compute linear index
    size_t ind = Index(n * (P(0, i) - min(0)) / (max(0) - min(0)));
    for (Index j = 1; j < P.rows(); ++j)
      ind = n * ind + Index(n * (P(j, i) - min(j)) / (max(j) - min(j)));
    if (boxes[ind] == false) {
      indices.push_back(i);
      boxes[ind] = true;
    }
  }
  indices.shrink_to_fit();
  return indices;
}
}  // namespace FMCA
#endif

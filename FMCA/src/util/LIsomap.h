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
#ifndef FMCA_UTIL_LISOMAP_H_
#define FMCA_UTIL_LISOMAP_H_

#include <iostream>
//
#include <Eigen/Eigenvalues>
//
#include "Macros.h"

namespace FMCA {

template <typename Graph>
Matrix LIsomap(const Graph &G, const Index M, const Index emb_dim,
               Scalar *nrg = nullptr) {
  using IndexType = typename Graph::IndexType;
  using ValueType = typename Graph::ValueType;
  const Index dim = emb_dim >= 1 ? (emb_dim <= M ? emb_dim : M) : 1;
  std::vector<IndexType> lm_ids = G.computeLandmarkNodes(M);
  std::vector<std::vector<ValueType>> D = G.partialDistanceMatrix(lm_ids);
  // generate and clean distance matrix of landmarks
  Matrix Dlm(M, M);
  for (Index j = 0; j < Dlm.cols(); ++j)
    for (Index i = 0; i < Dlm.rows(); ++i) Dlm(i, j) = D[i][lm_ids[j]];
  Scalar maxc = 0;
  for (Index j = 0; j < Dlm.cols(); ++j)
    for (Index i = 0; i < Dlm.rows(); ++i)
      if (Dlm(i, j) < FMCA_INF) maxc = maxc < Dlm(i, j) ? Dlm(i, j) : maxc;
  for (Index j = 0; j < Dlm.cols(); ++j)
    for (Index i = 0; i < Dlm.rows(); ++i)
      Dlm(i, j) = Dlm(i, j) < 1e1 * maxc ? Dlm(i, j) : 1e1 * maxc;

  const Vector ones = Vector::Ones(Dlm.rows());
  const auto H = Matrix::Identity(Dlm.rows(), Dlm.rows()) -
                 (1. / Dlm.rows()) * (ones * ones.transpose());
  const Matrix B = -0.5 * H * (Dlm.array().square().matrix()) * H;
  Eigen::SelfAdjointEigenSolver<FMCA::Matrix> es(B);
  const Index nneg = (es.eigenvalues().array() < 0).count();
  const Index npos = D.rows() - nneg;
  const Index cutr = npos >= dim ? dim : npos;
  const Scalar total_energy = es.eigenvalues().tail(npos).sum();
  const Scalar acc_energy = es.eigenvalues().tail(cutr).sum();
  if (nrg != nullptr) *nrg = (total_energy - acc_energy) / total_energy;

  return ?;
}
}  // namespace FMCA
#endif

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
  // clean distance matrix removing inf
  Scalar maxc = 0;
#pragma omp parallel for reduction(max : maxc)
  for (Index i = 0; i < D.size(); ++i)
    for (Index j = 0; j < D[i].size(); ++j)
      if (D[i][j] < std::numeric_limits<ValueType>::infinity())
        maxc = maxc < D[i][j] ? D[i][j] : maxc;
#pragma omp parallel for
  for (Index i = 0; i < D.size(); ++i)
    for (Index j = 0; j < D[i].size(); ++j)
      D[i][j] = D[i][j] < 1e1 * maxc ? D[i][j] : 1e1 * maxc;
  // create landmark distance matrix
  Matrix Dlm(M, M);
  for (Index j = 0; j < Dlm.cols(); ++j)
    for (Index i = 0; i < Dlm.rows(); ++i) Dlm(i, j) = D[i][lm_ids[j]];
  const Vector ones = Vector::Ones(Dlm.rows());
  const auto H = Matrix::Identity(Dlm.rows(), Dlm.rows()) -
                 (1. / Dlm.rows()) * (ones * ones.transpose());
  const Matrix B = -0.5 * H * (Dlm.array().square().matrix()) * H;
  Eigen::SelfAdjointEigenSolver<FMCA::Matrix> es(B);
  const Index nneg = (es.eigenvalues().array() < 0).count();
  const Index npos = Dlm.rows() - nneg;
  const Index cutr = npos >= dim ? dim : npos;
  const Scalar total_energy = es.eigenvalues().tail(npos).sum();
  const Scalar acc_energy = es.eigenvalues().tail(cutr).sum();
  if (nrg != nullptr) *nrg = (total_energy - acc_energy) / total_energy;
  // construct embedding, using the exact embedding for landmarks
  Matrix P(dim, G.nnodes());
  const Matrix E = es.eigenvectors().rightCols(cutr).transpose();
  const Matrix S = es.eigenvalues().tail(cutr).cwiseSqrt().asDiagonal();
  const Matrix invS = (1. / es.eigenvalues().tail(cutr).cwiseSqrt().array())
                          .matrix()
                          .asDiagonal();

  // set landmarks first
  for (Index i = 0; i < lm_ids.size(); ++i) P.col(lm_ids[i]) = S * E.col(i);
  // now set the rest, create landmark flag to know where to put them
  std::vector<bool> is_lm(G.nnodes(), false);
  for (Index i = 0; i < lm_ids.size(); ++i) is_lm[lm_ids[i]] = true;
  Vector mean = (Dlm.array().square().matrix()).colwise().mean();
#pragma omp parallel for schedule(dynamic)
  for (Index i = 0; i < P.cols(); ++i)
    if (not is_lm[i]) {
      Vector dist2(M);
      for (Index j = 0; j < M; ++j) dist2(j) = D[j][i] * D[j][i];
      P.col(i) = 0.5 * invS * E * (mean - dist2);
    }
  return P;
}
}  // namespace FMCA
#endif

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
#ifndef FMCA_UTIL_MDS_H_
#define FMCA_UTIL_MDS_H_

#include <iostream>
//
#include <Eigen/Eigenvalues>
//
#include "Macros.h"

namespace FMCA {

Matrix MDS(const Matrix &D, const Index emb_dim) {
  const Scalar dim =
      emb_dim >= 1 ? (emb_dim <= D.rows() ? emb_dim : D.rows()) : 1;
  const Vector ones = Vector::Ones(D.rows());
  Matrix D_clean = D;
  Scalar maxc = 0;
  for (Index j = 0; j < D.cols(); ++j)
    for (Index i = 0; i < D.rows(); ++i)
      if (D(i, j) < FMCA_INF) maxc = maxc < D(i, j) ? D(i, j) : maxc;
  for (Index j = 0; j < D.cols(); ++j)
    for (Index i = 0; i < D.rows(); ++i)
      D_clean(i, j) = D(i, j) < 1e1 * maxc ? D(i, j) : 1e1 * maxc;
  const auto H = Matrix::Identity(D.rows(), D.rows()) -
                 (1. / D.rows()) * (ones * ones.transpose());
  const Matrix B = -0.5 * H * (D_clean.array().square().matrix()) * H;
  Eigen::SelfAdjointEigenSolver<FMCA::Matrix> es(B);
  const Index nneg = (es.eigenvalues().array() < 0).count();
  const Index npos = D.rows() - nneg;
  const Index cutr = npos >= dim ? dim : npos;
  const Scalar total_energy = es.eigenvalues().tail(npos).sum();
  const Scalar acc_energy = es.eigenvalues().tail(cutr).sum();
#ifdef FMCA_VERBOSE
  std::cout << "msize: " << D.rows() << " cutr: " << cutr
            << " relative lost energy: "
            << (total_energy - acc_energy) / total_energy << std::endl;
#endif
  return es.eigenvalues().tail(cutr).cwiseSqrt().asDiagonal() *
         es.eigenvectors().rightCols(cutr).transpose();
}
}  // namespace FMCA
#endif

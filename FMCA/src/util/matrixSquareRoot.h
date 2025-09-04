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
#ifndef FMCA_UTIL_MATRIXSQUAREROOT_H_
#define FMCA_UTIL_MATRIXSQUAREROOT_H_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "Macros.h"

namespace FMCA {

template <typename T>
Vector matrixSquareRoot(const T& mat, const Vector& x,
                        const Index ksize = 100) {
  Matrix Q(mat.rows(), ksize);
  Q.setZero();
  Q.col(0) = x;
  Q.col(0).normalize();
  // determine orthogonal basis of Krylov subspace using Gram Schmidt
  for (Index i = 1; i < ksize; ++i) {
    Q.col(i) = mat * Q.col(i - 1);
    for (Index j = 0; j < 2; ++j) {
      const Vector QTv = Q.leftCols(i).transpose() * Q.col(i);
      Q.col(i) = Q.col(i) - Q.leftCols(i) * QTv;
    }
    const Scalar nrm = Q.col(i).norm();
    if (nrm < 100 * FMCA_ZERO_TOLERANCE) {
      Q.conservativeResize(Q.rows(), i);
      break;
    } else {
      Scalar scal = 1. / nrm;
      Q.col(i) *= scal;
    }
  }
  Matrix QTQ = Q.transpose() * Q;
  std::cout << "orthogonality error: "
            << (QTQ - Matrix::Identity(ksize, ksize)).norm() / std::sqrt(ksize)
            << std::endl;
  Matrix QTTQ = Q.transpose() * (mat * Q).eval();

  Eigen::SelfAdjointEigenSolver<Matrix> es(QTTQ);
  Vector evals = es.eigenvalues();
  for (FMCA::Index i = 0; i < evals.size(); ++i)
    evals(i) = evals(i) > 0 ? std::sqrt(evals(i)) : 0;
  std::cout << "energy loss: "
            << (evals.array().square().matrix() - es.eigenvalues()).norm() /
                   es.eigenvalues().norm()
            << std::endl;
  Vector y = Q.transpose() * x;
  y = es.eigenvectors().transpose() * y;
  y = evals.asDiagonal() * y;
  y = es.eigenvectors() * y;
  return Q * y;
}

}  // namespace FMCA

#endif

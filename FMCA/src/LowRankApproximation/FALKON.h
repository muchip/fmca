// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_LOWRANKAPPROXIMATION_FALKON_H_
#define FMCA_LOWRANKAPPROXIMATION_FALKON_H_
#include <random>

namespace FMCA {
class FALKON {
 public:
  FALKON() {
    L_.resize(0, 0);
    indices_.resize(0);
  }

  FALKON(const CovarianceKernel &ker, const Matrix &P, const Index rank)
      : rank_(rank) {
    L_.resize(0, 0);
    indices_.resize(0);
    compute(ker, P, rank);
  }
#if 0
function alpha = FALKON(X, C, Y, KernelMatrix, lambda, t)
  n = size(X,1); M = size(C,1); KMM = KernelMatrix(C,C);
  T = chol(KMM + eps*M*eye(M)); %MATLAB Cholesky upper triangular
  A = chol(T*T’/M + lambda*eye(M));
  function w = KnM_times_vector(u, v)
    w = zeros(M,1); ms = ceil(linspace(0, n, ceil(n/M)+1));
    for i=1:ceil(n/M)
        Kr = KernelMatrix( X(ms(i)+1:ms(i+1),:), C );
        w = w + Kr’*(Kr*u + v(ms(i)+1:ms(i+1),:));
    end
  end
  BHB = @(u) A’\(T’\(KnM_times_vector(T\(A\u), zeros(n,1))/n) + lambda*(A\u));
  r = A’\(T’\KnM_times_vector(zeros(M,1), Y/n));
  alpha = T\(A\conjgrad(BHB, r, t));
end
#endif
  void init(const CovarianceKernel &ker, const Matrix &P, const Index M,
            const Scalar lambda = 0.) {
    const Index dim = P.cols();
    const Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    std::mt19937 mtwister_;
    assert(M <= max_cols && "Nystrom rank too high");
    indices_.resize(dim);
    std::iota(indices_.begin(), indices_.end(), 0);
    std::shuffle(indices_.begin(), indices_.end(), mtwister_);
    indices_.conservativeResize(M);
    C_.resize(P.rows(), M);
    for (Index i = 0; i < M; ++i) C_.col(i) = P.col(indices_(i));
    KPC_ = ker.eval(P, C_);
    KCC_ = ker.eval(C_, C_);
    T_ = KCC_.llt().matrixL().transpose();
    A_ = (1. / M * T_ * T_.transpose() + lambda * FMCA::Matrix::Identity(M, M))
             .llt()
             .matrixL()
             .transpose();
    invT_ = T_.inverse();
    invA_ = A_.inverse();
    return;
  }

  const Matrix precmatvec(const Matrix &rhs) {}

  const iVector &indices() { return indices_; }

 private:
  // member variables
  Matrix KPC_;
  Matrix KCC_;
  Matrix C_;
  Matrix T_;
  Matrix T_;
  Matrix invT_;
  Matrix invA_;
  iVector indices_;
  Index rank_;
  // we cap the maximum matrix size at 8GB
  const Index max_size_ = Index(1e9);
};
}  // namespace FMCA
#endif

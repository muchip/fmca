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

namespace FMCA {
class FALKON {
 public:
  FALKON() {}

  FALKON(const CovarianceKernel &ker, const Matrix &P, const Index rank,
         const Scalar lambda = 0.)
      : lambda_(lambda) {
    init(ker, P, rank, lambda);
  }

  /**
   *  \brief KnM_times_vector implementation from
   *  [Rudi, Carratino, Rosasco. FALKON: An Optimal Large Scale Kernel Method]
   *  function w = KnM_times_vector(u, v)
   *    w = zeros(M,1);
   *    ms = ceil(linspace(0, n, ceil(n/M)+1));
   *    for i=1:ceil(n/M)
   *      Kr = KernelMatrix( X(ms(i)+1:ms(i+1),:), C );
   *      w = w + Kr’*(Kr*u + v(ms(i)+1:ms(i+1),:));
   *    end
   *  end
   **/
  Matrix KTKTimesVector(const Matrix &u, const Matrix &v) {
    return KPC_.transpose() * ((KPC_ * u).eval() + v);
  }

  /**
   *  \brief BHB implementation from
   *  [Rudi, Carratino, Rosasco. FALKON: An Optimal Large Scale Kernel Method]
   *  BTB = @(u) A'\(T'\(KnM_times_vector(T\(A\u), zeros(n,1))/n)+lambda*(A\u));
   * **/
  Matrix BTBTimesVector(const Matrix &u) {
    const Scalar invn = 1. / KPC_.rows();
    const Matrix zero = FMCA::Matrix::Zero(KPC_.rows(), u.cols());
    return invA_.transpose() *
           (invn * invT_.transpose() * KTKTimesVector(invT_ * invA_ * u, zero) +
            lambda_ * invA_ * u);
  }

  /**
   *  \brief initialization implementation from
   *  [Rudi, Carratino, Rosasco. FALKON: An Optimal Large Scale Kernel Method]
   *  T = chol(KMM + eps*M*eye(M));
   *  A = chol(T*T’/M + lambda*eye(M));
   **/
  void init(const CovarianceKernel &ker, const Matrix &P, const Index M,
            const Scalar lambda = 0.) {
    const Index dim = P.cols();
    const Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    const Matrix I = FMCA::Matrix::Identity(M, M);
    lambda_ = lambda;
    assert(M <= max_cols && "Nystrom rank too high");
    // randomly select Nystrom centers
    {
      std::mt19937 mtwister_;
      indices_.resize(dim);
      std::iota(indices_.begin(), indices_.end(), 0);
      std::shuffle(indices_.begin(), indices_.end(), mtwister_);
      indices_.conservativeResize(M);
      C_.resize(P.rows(), M);
      for (Index i = 0; i < M; ++i) C_.col(i) = P.col(indices_(i));
    }
    // explicitly set up KnM assuming that we are not memory bound
    KPC_ = ker.eval(P, C_);
    // set up KMM
    KCC_ = ker.eval(C_, C_);
    // T = chol(KMM + eps*M*eye(M));
    T_ = (KCC_ + FMCA_ZERO_TOLERANCE * M * I).llt().matrixL().transpose();
    // A = chol(T*T’/M + lambda*eye(M));
    A_ = (T_ * T_.transpose() / M + lambda * I).llt().matrixL().transpose();
    // invert matrices explicitly (still cost M^3)
    invT_ = T_.inverse();
    invA_ = A_.inverse();
    return;
  }

  /**
   *  \brief CG solver implementation from
   *  [Rudi, Carratino, Rosasco. FALKON: An Optimal Large Scale Kernel Method]
   *  r = A'\(T'\KnM_times_vector(zeros(M,1), Y/n));
   *  alpha = T\(A\conjgrad(BHB, r, t));
   **/
  Vector computeAlpha(const Vector &Y, const Index t = 10) {
    const Scalar invn = 1. / KPC_.rows();
    const Matrix zero = FMCA::Matrix::Zero(indices().size(), 1);
    // r = A'\(T'\KnM_times_vector(zeros(M,1), Y/n));
    const Vector rhs =
        invA_.transpose() * invT_.transpose() * KTKTimesVector(zero, invn * Y);
    Vector x(rhs.size());
    const Scalar rhsnorm = rhs.norm();
    // perform CG iterations
    {
      Vector res = rhs;
      Vector p = rhs;
      Vector res_old;
      Scalar rtr = res.dot(res);
      Scalar rtr_old;
      Scalar err = std::sqrt(rtr) / rhsnorm;
      x.setZero();
      Index k = 0;
      for (; k < t && err > FMCA_ZERO_TOLERANCE; ++k) {
        const Vector Ap = BTBTimesVector(p);
        const Scalar alpha = rtr / p.dot(Ap);
        x += alpha * p;
        res_old = res;
        rtr_old = rtr;
        res -= alpha * Ap;
        rtr = res.dot(res);
        const Scalar beta = rtr / rtr_old;
        p = res + beta * p;
        err = std::sqrt(rtr) / rhsnorm;
      }
      std::cout << "CG iterations: " << k << " relative residual: " << err
                << std::endl;
    }
    // alpha = T\(A\conjgrad(BHB, r, t));
    return invT_ * invA_ * x;
  }

  const iVector &indices() const { return indices_; }
  const Matrix &matrixC() const { return C_; }
  const Matrix &matrixKPC() const { return KPC_; }

 private:
  // member variables
  Matrix KPC_;
  Matrix KCC_;
  Matrix C_;
  Matrix T_;
  Matrix A_;
  Matrix invT_;
  Matrix invA_;
  iVector indices_;
  Scalar lambda_;
  // we cap the maximum matrix size at 8GB
  const Index max_size_ = Index(1e9);
};
}  // namespace FMCA
#endif

// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//

#ifndef FMCA_COLLOCATION_PIKLSOLVER_H_
#define FMCA_COLLOCATION_PIKLSOLVER_H_

namespace FMCA {

/**
 *  \ingroup Collocation
 *  \brief Physics-informed kernel learning: instead of the square collocation
 *         system G z = b, the regularised least-squares problem
 *
 *      min_z  w1^2 ||A zI + B zB - fI||^2 + w2^2 ||C zI + D zB - fB||^2
 *             + lambda2 ||z||^2,   w1^2 = 1/nI,  w2^2 = lambda1/nB,
 *
 *  solved through its normal equations N z = c, N = G^T W G + lambda2 I,
 *  c = G^T W b.
 *
 *  Solver: deflated PCG (variant A-DEF2 of Tang, Nabben, Vuik, Erlangga '09).
 *  The preconditioner P^{-1} = G^{-1} W^{-1} G^{-T} inverts the data term
 *  exactly, so P^{-1} N = I + lambda2 P^{-1} has eigenvalues 1 + lambda2/mu,
 *  mu the eigenvalues of G^T W G.  The only directions where this is far from
 *  one are those where G^T W G is small, and these are the nB discrete
 *  harmonic coefficients
 *
 *      Z = [ -Y ; I ],   Y = A^{-1} B,   for which   G Z = [ 0 ; S ].
 *
 *  They are removed by deflation with the coarse matrix
 *
 *      E = Z^T N Z = w2^2 S^T S + lambda2 (I + Y^T Y),      Q = Z E^{-1} Z^T,
 *
 *  i.e. the regularised problem is solved exactly on Z and PCG only works on
 *  the remaining directions, where the data term dominates lambda2.
 *
 *  \note setRegularization() must be called before compute(): the coarse
 *        matrix E depends on lambda1 and lambda2.
 **/
class PIKLSolver {
 public:
  PIKLSolver() noexcept
      : lambda1_(1.),
        lambda2_(0.),
        tol_(1e-10),
        maxit_(200),
        iterations_(0),
        residual_(0) {}

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief lambda1 weights the boundary condition against the differential
   *         equation, lambda2 the regulariser.  lambda2 = 0 reproduces the
   *         collocation solution.
   **/
  void setRegularization(Scalar lambda1, Scalar lambda2) {
    lambda1_ = lambda1 > 0 ? lambda1 : 1.;
    lambda2_ = lambda2 >= 0 ? lambda2 : 0;
    return;
  }

  /**
   *  \brief Tolerance and iteration cap of the outer preconditioned CG.
   **/
  void setParameters(Scalar tol, Index maxit) {
    tol_ = tol > 0 ? tol : 1e-10;
    maxit_ = maxit > 0 ? maxit : 200;
    return;
  }

  /**
   *  \brief Tolerance, cap and nugget of the interior CG inside G^{-1} and
   *         G^{-T}.  They have to be tight: the preconditioner is a fixed
   *         linear operator only if these solves are accurate, and otherwise
   *         the outer iteration loses conjugacy and stagnates.
   **/
  void setInteriorParameters(Scalar tol, Index maxit, Scalar nugget = 1e-10) {
    schur_.setParameters(tol, maxit, nugget);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Factorises G and G^T, which share their interior block, and the
   *         coarse matrix E of the deflation space.
   **/
  void compute(const SparseMatrix& A, const SparseMatrix& B,
               const SparseMatrix& C, const SparseMatrix& D) {
    A_ = &A;
    B_ = &B;
    C_ = &C;
    D_ = &D;
    nI_ = A.rows();
    nB_ = D.rows();
    w1s_ = 1. / Scalar(nI_);
    w2s_ = lambda1_ / Scalar(nB_);
    schur_.compute(A, B, C, D);
    // E = Z^T N Z = w2^2 S^T S + lambda2 (I + Y^T Y), dense nB x nB, SPD
    const Matrix& Y = schur_.YB();
    const Matrix& S = schur_.S();
    Matrix E = w2s_ * (S.transpose() * S) + lambda2_ * (Y.transpose() * Y);
    E.diagonal().array() += lambda2_;
    lltE_.compute(E);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Solves the regularised normal equations by deflated PCG.
   *
   *  Standard PCG on N z = c with the preconditioner
   *      (I - Q N) P^{-1} + Q
   *  and the starting vector z0 = Q c (A-DEF2).  Per iteration: two products
   *  with N, one application of P^{-1} (three interior CG solves) and one of
   *  Q (two products with Y and one small triangular solve).
   **/
  Vector solve(const Vector& fI, const Vector& fB) {
    const Index n = nI_ + nB_;
    Vector c(n);  // the right hand side G^T W b
    c.head(nI_) = A_->transpose() * (w1s_ * fI) + C_->transpose() * (w2s_ * fB);
    c.tail(nB_) = B_->transpose() * (w1s_ * fI) + D_->transpose() * (w2s_ * fB);
    const Scalar c0 = c.norm();
    iterations_ = 0;
    residual_ = 0;
    if (c0 <= 0) return Vector::Zero(n);
    Vector z = applyQ(c);  // exact on the deflation space
    Vector r = c - applyNormal(z);
    Vector p = applyDeflated(r);
    Vector d = p;
    Scalar rho = r.dot(p);
    while (iterations_ < maxit_) {
      residual_ = r.norm() / c0;
      if (residual_ < tol_) break;
      const Vector q = applyNormal(d);
      const Scalar alpha = rho / d.dot(q);
      z += alpha * d;
      r -= alpha * q;
      p = applyDeflated(r);
      const Scalar rho_new = r.dot(p);
      d = p + (rho_new / rho) * d;
      rho = rho_new;
      ++iterations_;
    }
    return z;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Previous solver: PCG with the preconditioner P^{-1} alone.  
  //
  // Vector solve(const Vector& fI, const Vector& fB) {
  //   const Index n = nI_ + nB_;
  //   Vector c(n);  // the right hand side G^T W b
  //   c.head(nI_) = A_->transpose() * (w1s_ * fI) + C_->transpose() * (w2s_ * fB);
  //   c.tail(nB_) = B_->transpose() * (w1s_ * fI) + D_->transpose() * (w2s_ * fB);
  //   const Scalar c0 = c.norm();
  //   Vector z = Vector::Zero(n);
  //   Vector r = c;
  //   Vector d = applyPinv(c);
  //   Scalar rho = r.dot(d);
  //   residual_ = 1.;
  //   iterations_ = 0;
  //   while (iterations_ < maxit_) {
  //     residual_ = r.norm() / c0;
  //     if (residual_ < tol_) break;
  //     const Vector q = applyNormal(d);
  //     const Scalar alpha = rho / d.dot(q);
  //     z += alpha * d;
  //     r -= alpha * q;
  //     const Vector v = applyPinv(r);
  //     const Scalar rho_new = r.dot(v);
  //     d = v + (rho_new / rho) * d;
  //     rho = rho_new;
  //     ++iterations_;
  //   }
  //   return z;
  // }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  const Index iterations() const { return iterations_; }
  const Scalar residual() const { return residual_; }
  const Index interiorIterations() const { return schur_.interiorIterations(); }
  Scalar conditionEstimate(Index k = 100) const {
    return schur_.conditionEstimate(k);
  }

 private:
  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief (G^T W G + lambda2 I) z
   **/
  Vector applyNormal(const Vector& z) {
    const Vector tI = w1s_ * ((*A_) * z.head(nI_) + (*B_) * z.tail(nB_));
    const Vector tB = w2s_ * ((*C_) * z.head(nI_) + (*D_) * z.tail(nB_));
    Vector r(nI_ + nB_);
    r.head(nI_) = A_->transpose() * tI + C_->transpose() * tB;
    r.tail(nB_) = B_->transpose() * tI + D_->transpose() * tB;
    if (lambda2_ > 0) r += lambda2_ * z;
    return r;
  }

  /**
   *  \brief (G^T W G)^{-1} r = G^{-1} W^{-1} G^{-T} r, two Schur solves.
   **/
  Vector applyPinv(const Vector& r) {
    Vector y = schur_.solveTransposed(Vector(r.head(nI_)), Vector(r.tail(nB_)));
    y.head(nI_) /= w1s_;
    y.tail(nB_) /= w2s_;
    return schur_.solve(Vector(y.head(nI_)), Vector(y.tail(nB_)));
  }

  /**
   *  \brief Q v = Z E^{-1} Z^T v with Z = [-Y; I]:  Z^T v = vB - Y^T vI and
   *         Z w = [-Y w; w].
   **/
  Vector applyQ(const Vector& v) {
    const Matrix& Y = schur_.YB();
    const Vector w =
        lltE_.solve(Vector(v.tail(nB_) - Y.transpose() * v.head(nI_)));
    Vector q(nI_ + nB_);
    q.head(nI_) = -(Y * w);
    q.tail(nB_) = w;
    return q;
  }

  /**
   *  \brief The deflated preconditioner ((I - Q N) P^{-1} + Q) r, written as
   *         y + Q (r - N y) with y = P^{-1} r.
   **/
  Vector applyDeflated(const Vector& r) {
    const Vector y = applyPinv(r);
    return y + applyQ(Vector(r - applyNormal(y)));
  }

  SchurSolver schur_;
  Eigen::LLT<Matrix> lltE_;
  const SparseMatrix* A_;
  const SparseMatrix* B_;
  const SparseMatrix* C_;
  const SparseMatrix* D_;
  Scalar lambda1_;
  Scalar lambda2_;
  Scalar w1s_;
  Scalar w2s_;
  Scalar tol_;
  Index maxit_;
  Index nI_;
  Index nB_;
  Index iterations_;
  Scalar residual_;
};
}  // namespace FMCA

#endif

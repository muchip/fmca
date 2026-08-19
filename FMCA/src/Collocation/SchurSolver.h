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

#ifndef FMCA_COLLOCATION_SCHURSOLVER_H_
#define FMCA_COLLOCATION_SCHURSOLVER_H_

namespace FMCA {

/**
 *  \ingroup Collocation
 *  \brief Block elimination of the collocation system
 *
 *      [ A  B ] [ zI ]   [ fI ]
 *      [ C  D ] [ zB ] = [ fB ].
 *
 *  The block -A is symmetric positive definite, so the conjugate gradient
 *  method applies to it.  Eliminating zI leaves the small boundary system
 *  (D - C A^{-1} B) zB = fB - C A^{-1} fI, which is factorised densely.  Both
 *  A^{-1}B and that factorisation are computed ONCE in compute(), so every
 *  later solve() costs a single interior CG solve.
 *
 *  \note compute() keeps pointers to the blocks handed to it. They must stay
 *        alive for as long as the solver is used.
 **/
class SchurSolver {
 public:
  using CG = Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                                      Eigen::DiagonalPreconditioner<Scalar>>;

  SchurSolver() noexcept
      : tol_(1e-6), maxit_(200), nugget_(1e-10), iterations_(0) {}

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Tolerance, iteration cap and Tikhonov nugget of the interior CG.
   **/
  void setParameters(Scalar tol, Index maxit, Scalar nugget = 1e-10) {
    tol_ = tol > 0 ? tol : 1e-6;
    maxit_ = maxit > 0 ? maxit : 200;
    nugget_ = nugget >= 0 ? nugget : 0;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Sets up the interior CG on -A + nugget * I, computes A^{-1}B and
   *         factorises the boundary Schur complement.
   **/
  void compute(const SparseMatrix& A, const SparseMatrix& B,
               const SparseMatrix& C, const SparseMatrix& D,
               bool with_transpose = false) {
    B_ = &B;
    C_ = &C;
    nI_ = A.rows();
    nB_ = D.rows();
    Aspd_ = -A;  // -A is symmetric positive definite, the nugget guards the CG
    SparseMatrix Id(nI_, nI_);
    Id.setIdentity();
    Aspd_ += nugget_ * Id;
    cg_.setTolerance(tol_);
    cg_.setMaxIterations(maxit_);
    cg_.compute(Aspd_);
    YB_ = interiorSolve(Matrix(B));
    lu_.compute(Matrix(D) - Matrix(C) * YB_);
    if (with_transpose) {
      YC_ = interiorSolve(Matrix(C.transpose()));
      luT_.compute(Matrix(D.transpose()) - Matrix(B.transpose()) * YC_);
    }
    iterations_ = Index(cg_.iterations());
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Solves G z = [fI; fB], at the cost of one interior CG solve.
   **/
  Vector solve(const Vector& fI, const Vector& fB) {
    const Vector z = interiorSolve(fI);
    const Vector zB = lu_.solve(fB - (*C_) * z);
    Vector x(nI_ + nB_);
    x.head(nI_) = z - YB_ * zB;
    x.tail(nB_) = zB;
    iterations_ = Index(cg_.iterations());
    return x;
  }

  /**
   *  \brief Solves G^T z = [fI; fB].  Requires compute(..., true).
   **/
  Vector solveTransposed(const Vector& fI, const Vector& fB) {
    const Vector z = interiorSolve(fI);
    const Vector zB = luT_.solve(fB - B_->transpose() * z);
    Vector x(nI_ + nB_);
    x.head(nI_) = z - YC_ * zB;
    x.tail(nB_) = zB;
    return x;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief 2-norm condition number of the interior block, from the extreme
   *         Ritz values of k Lanczos steps with full reorthogonalisation.  Only
   *         products with the block are used.  This is an estimate meant for a
   *         condition-versus-N plot, not a certified bound, and it costs k
   *         matrix-vector products.
   **/
  Scalar conditionEstimate(Index k = 100) const {
    const Index n = Aspd_.rows();
    if (n <= 1) return 1.;
    k = k < n ? k : n;
    std::vector<Vector> V;
    std::vector<Scalar> diag, offd;
    Vector v = Vector::Ones(n).normalized();
    Vector vprev = Vector::Zero(n);
    Scalar beta = 0;
    for (Index j = 0; j < k; ++j) {
      Vector w = Aspd_ * v;
      if (j > 0) w -= beta * vprev;
      const Scalar alpha = v.dot(w);
      w -= alpha * v;
      V.push_back(v);
      for (const Vector& q : V) w -= q.dot(w) * q;  // full reorthogonalisation
      diag.push_back(alpha);
      beta = w.norm();
      if (beta < 10 * FMCA_ZERO_TOLERANCE) break;
      offd.push_back(beta);
      vprev = v;
      v = w / beta;
    }
    const Index m = diag.size();
    Matrix T = Matrix::Zero(m, m);
    for (Index j = 0; j < m; ++j) {
      T(j, j) = diag[j];
      if (j + 1 < m) {
        T(j, j + 1) = offd[j];
        T(j + 1, j) = offd[j];
      }
    }
    const SelfAdjointEigenSolver es(T);
    const Scalar lmin = es.eigenvalues().minCoeff();
    const Scalar lmax = es.eigenvalues().maxCoeff();
    const Scalar floor = FMCA_ZERO_TOLERANCE * lmax;
    return lmax / (lmin > floor ? lmin : floor);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  const Index iterations() const { return iterations_; }

 private:
  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief A^{-1}V.  Since Aspd_ = -A + nugget * I, this solves Aspd_ X = -V.
   **/
  Matrix interiorSolve(const Matrix& V) { return cg_.solve(Matrix(-V)); }

  CG cg_;
  SparseMatrix Aspd_;
  Matrix YB_;
  Matrix YC_;
  Eigen::PartialPivLU<Matrix> lu_;
  Eigen::PartialPivLU<Matrix> luT_;
  const SparseMatrix* B_;
  const SparseMatrix* C_;
  Scalar tol_;
  Index maxit_;
  Scalar nugget_;
  Index nI_;
  Index nB_;
  Index iterations_;
};
}  // namespace FMCA

#endif

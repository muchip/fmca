// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_LOWRANKAPPROXIMATION_PIVOTEDCHOLESKY_H_
#define FMCA_LOWRANKAPPROXIMATION_PIVOTEDCHOLESKY_H_

namespace FMCA {
class PivotedCholesky {
 public:
  PivotedCholesky() {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    tol_ = 0;
    dim_ = 0;
    max_cols_ = 0;
    info_ = 0;
    return;
  }
  struct KernelMatrixWrapper {
    KernelMatrixWrapper(const CovarianceKernel &ker, Matrix &&P) = delete;
    KernelMatrixWrapper(const CovarianceKernel &ker, const Matrix &P)
        : ker_(ker), P_(P) {}
    Vector diagonal() const {
      Vector retval(P_.cols());
      for (Index i = 0; i < retval.size(); ++i)
        retval(i) = (ker_.eval(P_.col(i), P_.col(i)))(0, 0);
      return retval;
    }
    Vector col(Index i) const { return ker_.eval(P_, P_.col(i)); }

    Index rows() const { return P_.cols(); }
    Index cols() const { return P_.cols(); }
    // members
    const CovarianceKernel &ker_;
    const Matrix &P_;
  };

  PivotedCholesky(const CovarianceKernel &ker, const Matrix &P,
                  Scalar tol = 1e-3)
      : tol_(tol) {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    dim_ = P.cols();
    max_cols_ = max_size_ / dim_ > dim_ ? dim_ : max_size_ / dim_;
    compute(ker, P, tol);
    return;
  }

  // Classical variant of the diagonally pivoted Cholesky decomposition
  template <typename T>
  static int PgreedyPCD(const T &K, Matrix *L, iVector *idcs, Scalar tol = 1e-3,
                        Index max_cols = 1000) {
    Vector D = K.diagonal();
    Scalar tr = D.sum();
    Index pivot = 0;
    Index step = 0;
    if (D.minCoeff() < 0) return 1;
    ////////////////////////////////////////////////////////////////////////////
    max_cols = max_cols > K.cols() ? K.cols() : max_cols;
    L->resize(K.rows(), max_cols);
    idcs->resize(max_cols);
    // we guarantee the error tr(K-LL^T)/tr(K) < tol
    tol *= tr;
    while ((step < max_cols) && (tol < tr)) {
      D.maxCoeff(&pivot);
      (*idcs)[step] = pivot;
      const Vector col =
          1. / std::sqrt(D(pivot)) *
          (K.col(pivot) -
           L->leftCols(step) * L->row(pivot).head(step).transpose());
      L->col(step) = col;
      D.array() -= col.array().square();
      if (D.minCoeff() < -FMCA_ZERO_TOLERANCE) {
        L->conservativeResize(L->rows(), step);
        idcs->conservativeResize(step);
        return 2;
      }
      D = D.cwiseMax(0);
      // compute the trace of the Schur complement
      tr = D.sum();
      ++step;
    }
    // crop L, indices to their actual size
    L->conservativeResize(L->rows(), step);
    idcs->conservativeResize(step);
    return 0;
  }

  // Online QR variant of the diagonally pivoted Cholesky decomposition
  template <typename T>
  static int PgreedyPCDQR(const T &K, Matrix *Q, Matrix *R, iVector *idcs,
                          Scalar tol = 1e-3, Index max_cols = 1000) {
    Vector D = K.diagonal();
    if (D.minCoeff() < 0) return 1;
    max_cols = max_cols > K.cols() ? K.cols() : max_cols;
    Index pivot = 0;
    Scalar tr = 0;
    Q->resize(K.rows(), max_cols);
    R->resize(max_cols, max_cols);
    Q->setZero();
    R->setZero();
    idcs->resize(max_cols);
    tr = D.sum();
    // we guarantee the error tr(K-LL^T)/tr(K) < tol
    tol *= tr;
    // perform pivoted Cholesky decomposition
    Index step = 0;
    Index qstep = 0;
    while ((step < max_cols) && (tol < tr)) {
      D.maxCoeff(&pivot);
      (*idcs)[step] = pivot;
      const Scalar scal = 1. / std::sqrt(D(pivot));
      const Vector col = K.col(pivot);
      const Vector row = R->block(0, 0, qstep, step).transpose() *
                         Q->row(pivot).head(qstep).transpose();
      const Vector l =
          scal * (col - Q->leftCols(qstep) *
                            (R->block(0, 0, qstep, step) * row).eval());
      Vector q = l;
      Vector r(qstep);
      r.setZero();
      if (qstep)
        for (Index i = 0; i < 2; ++i) {
          const Vector cc = Q->leftCols(qstep).transpose() * q;
          q = q - Q->leftCols(qstep) * cc;
          r += cc;
        }
      const Scalar rho = q.norm();
      if (rho > 1e-4 * l.norm()) {
        Q->col(qstep) = (1. / rho) * q;
        R->col(step).head(qstep) = r;
        (*R)(qstep, step) = rho;
        ++qstep;
      } else {
        R->col(step).head(qstep) = r;
      }

      D.array() -= l.array().square();
      if (D.minCoeff() < -FMCA_ZERO_TOLERANCE) {
        Q->conservativeResize(Q->rows(), std::min(qstep, step));
        R->conservativeResize(std::min(qstep, step), step);
        idcs->conservativeResize(step);
        return 2;
      }
      D = D.cwiseMax(0);
      // compute the trace of the Schur complement
      tr = D.sum();
      ++step;
    }
    // crop L, indices to their actual size
    Q->conservativeResize(Q->rows(), std::min(qstep, step));
    R->conservativeResize(std::min(qstep, step), step);
    idcs->conservativeResize(step);
    return 0;
  }

  // f-greedy variant of the pivoted Cholesky decomposition
  template <typename T>
  static int fgreedyPCD(const T &K, const Vector &f, Matrix *L, Matrix *B,
                        iVector *idcs, Scalar tol = 1e-3,
                        Index max_cols = 1000) {
    Vector D = K.diagonal();
    if (D.minCoeff() < 0) return 1;
    Index pivot = 0;
    Index step = 0;
    Vector r = f;
    Scalar resnorm = r.norm();
    ////////////////////////////////////////////////////////////////////////////
    max_cols = max_cols > K.cols() ? K.cols() : max_cols;
    L->resize(K.rows(), max_cols);
    B->resize(K.rows(), max_cols);
    idcs->resize(max_cols);
    // we guarantee the error tr(K-LL^T)/tr(K) < tol
    tol *= resnorm;
    while ((step < max_cols) && (resnorm > tol)) {
      r.cwiseAbs().maxCoeff(&pivot);
      (*idcs)[step] = pivot;
      if (D.minCoeff() < -FMCA_ZERO_TOLERANCE ||
          D(pivot) < FMCA_ZERO_TOLERANCE) {
        L->conservativeResize(L->rows(), step);
        B->conservativeResize(B->rows(), step);
        idcs->conservativeResize(step);
        return 2;
      }
      const Vector updL =
          L->leftCols(step) * L->row(pivot).head(step).transpose();
      const Vector updB =
          B->leftCols(step) * L->row(pivot).head(step).transpose();
      L->col(step) = 1. / std::sqrt(D(pivot)) * (K.col(pivot) - updL);
      B->col(step) =
          1. / std::sqrt(D(pivot)) * (Vector::Unit(K.rows(), pivot) - updB);
      D.array() -= L->col(step).array().square();
      r -= r.dot(B->col(step)) * L->col(step);
      // compute the residual
      resnorm = r.norm();
      D = D.cwiseMax(0);
      ++step;
    }
    // crop L, indices to their actual size
    L->conservativeResize(L->rows(), step);
    B->conservativeResize(B->rows(), step);
    idcs->conservativeResize(step);
    return 0;
  }

  void compute(const CovarianceKernel &ker, const Matrix &P,
               Scalar tol = 1e-3) {
    B_.resize(0, 0);
    dim_ = P.cols();
    max_cols_ = max_size_ / dim_ > dim_ ? dim_ : max_size_ / dim_;
    tol_ = tol;
    info_ =
        PgreedyPCD(KernelMatrixWrapper(ker, P), &L_, &indices_, tol, max_cols_);
    return;
  }

  void computeOMP(const CovarianceKernel &ker, const Matrix &P, const Vector &f,
                  Scalar tol = 1e-3) {
    dim_ = P.cols();
    max_cols_ = max_size_ / dim_ > dim_ ? dim_ : max_size_ / dim_;
    tol_ = tol;
    info_ = fgreedyPCD(KernelMatrixWrapper(ker, P), f, &L_, &B_, &indices_, tol,
                       max_cols_);
    return;
  }

  void computeFullPiv(const CovarianceKernel &ker, const Matrix &P,
                      Scalar tol = 1e-3) {
    const Index dim = P.cols();
    const Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    tol_ = tol;
    if (max_cols < dim) {
      info_ = 3;
      return;
    }
    SelfAdjointEigenSolver es;
    {
      Matrix K = ker.eval(P, P);
      es.compute(K);
      info_ = es.info();
      if (es.info() != Success) return;
    }
    Vector ev = es.eigenvalues().reverse();
    std::cout << "lambda min: " << ev.minCoeff() << " "
              << "lambda max: " << ev.maxCoeff();
    Scalar tr = ev.sum();
    Scalar cur_tr = 0;
    Index step = 0;
    while (tr - cur_tr > tol * tr) {
      cur_tr += ev(step);
      ++step;
    }
    std::cout << " step: " << step << std::endl;
    L_.resize(dim, step);
    for (auto i = 1; i <= step; ++i)
      L_.col(i - 1) = es.eigenvectors().col(dim - i);
    L_ = L_ * ev.head(step).cwiseSqrt().asDiagonal();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /*
   *    \brief computes the biorthogonal basis B of the Cholesky factor such
   *           that B^TL=I
   */
  void computeBiorthogonalBasis() {
    B_.resize(L_.rows(), L_.cols());
    B_.setZero();
    for (auto i = 0; i < indices_.size(); ++i) {
      B_(indices_(i), i) = 1;
      B_.col(i) -= B_.block(0, 0, B_.rows(), i) *
                   L_.row(indices_(i)).head(i).transpose();
      B_.col(i) /= L_(indices_(i), i);
    }
  }

  /*
   *    \brief computes the weights for the double orthogonal basis, i.e.
   *           UV, where V is the spectral basis of L^TL
   */
  Matrix spectralBasisWeights() {
    // compute spectral decomposition of L^TL
    Matrix C = matrixL().transpose() * matrixL();
    SelfAdjointEigenSolver es(C);
    Matrix matrixQ = es.eigenvectors();
    eigenvalues_ = es.eigenvalues();
    // sort the eigen basis such that the eigenvalues are decreasing
    for (auto i = 0; i < matrixQ.cols() / 2; ++i) {
      matrixQ.col(i).swap(matrixQ.col(matrixQ.cols() - 1 - i));
      const Scalar val = eigenvalues_(i);
      eigenvalues_(i) = eigenvalues_(eigenvalues_.size() - 1 - i);
      eigenvalues_(eigenvalues_.size() - 1 - i) = val;
    }
    // assemble the actual weights
    return matrixU() * matrixQ;
  }

  Matrix matrixU() const {
    Matrix U(B_.cols(), B_.cols());
    for (Index i = 0; i < U.rows(); ++i) U.row(i) = B_.row(indices_(i));
    return U;
  }

  const Matrix &matrixB() const { return B_; }
  const Matrix &matrixL() const { return L_; }
  const iVector &indices() const { return indices_; }
  const Vector &eigenvalues() const { return eigenvalues_; }

  const Scalar &tol() const { return tol_; }
  const Index &info() const { return info_; }

 private:
  // member variables
  Matrix L_;
  Matrix B_;
  Vector eigenvalues_;
  iVector indices_;
  Scalar tol_;
  const Index max_size_ = Index(1e9);
  Index info_;
  Index dim_;
  Index max_cols_;
  // we cap the maximum matrix size at 8GB
};
}  // namespace FMCA
#endif

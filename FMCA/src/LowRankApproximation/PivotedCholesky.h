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
  }

  PivotedCholesky(const CovarianceKernel &ker, const Matrix &P,
                  Scalar tol = 1e-3)
      : tol_(tol) {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    compute(ker, P, tol);
  }

  void compute(const CovarianceKernel &ker, const Matrix &P,
               Scalar tol = 1e-3) {
    const Index dim = P.cols();
    const Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    Vector D(dim);
    Index pivot = 0;
    Scalar tr = 0;
    L_.resize(dim, max_cols);
    indices_.resize(max_cols);
    tol_ = tol;
    // compute the diagonal and the trace
    for (auto i = 0; i < dim; ++i) {
      const Matrix wtf = ker.eval(P.col(i), P.col(i));
      D(i) = wtf(0, 0);
      if (D(i) < 0) {
        info_ = 1;
        return;
      }
    }
    tr = D.sum();
    // we guarantee the error tr(A-LL^T)/tr(A) < tol
    tol *= tr;
    // perform pivoted Cholesky decomposition
    std::cout << "N: " << dim << " max number of cols: " << max_cols
              << std::endl
              << "rel tol: " << tol << " initial trace: " << tr << std::endl;
    Index step = 0;
    while ((step < max_cols) && (tol < tr)) {
      D.maxCoeff(&pivot);
      indices_(step) = pivot;
      // get new column from C
      L_.col(step) = ker.eval(P, P.col(pivot));
      // update column with the current matrix Lmatrix_
      L_.col(step) -= L_.leftCols(step) * L_.row(pivot).head(step).transpose();
      if (L_(pivot, step) <= 0) {
        info_ = 2;
        std::cout << "breaking with non positive pivot\n";
        break;
      }
      L_.col(step) /= sqrt(L_(pivot, step));
      // update the diagonal and the trace
      D.array() -= L_.col(step).array().square();
      // compute the trace of the Schur complement
      tr = D.sum();
      ++step;
    }
    std::cout << "steps: " << step << " trace error: " << tr << std::endl;
    if (tr < 0)
      info_ = 2;
    else
      info_ = 0;
    // crop L, indices to their actual size
    L_.conservativeResize(dim, step);
    indices_.conservativeResize(step);
    return;
  }

  void computeOMP(const CovarianceKernel &ker, const Matrix &P, const Vector &f,
                  Scalar tol = 1e-3) {
    const Index dim = P.cols();
    const Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    Vector D(dim);
    Vector r = f;
    Index pivot = 0;
    Scalar resnorm = 0;
    L_.resize(dim, max_cols);
    indices_.resize(max_cols);
    tol_ = tol;
    // compute the diagonal and the trace
    for (auto i = 0; i < dim; ++i) {
      const Matrix wtf = ker.eval(P.col(i), P.col(i));
      D(i) = wtf(0, 0);
      if (D(i) < 0) {
        info_ = 1;
        return;
      }
    }
    resnorm = r.norm();
    tol *= resnorm;
    // perform pivoted Cholesky decomposition
    std::cout << "N: " << dim << " max number of cols: " << max_cols
              << std::endl
              << "rel tol: " << tol << " initial residual: " << resnorm
              << std::endl;
    Index step = 0;
    while ((step < max_cols) && resnorm > tol) {
      r.cwiseAbs().maxCoeff(&pivot);
      indices_(step) = pivot;
      // get new column from C
      L_.col(step) = ker.eval(P, P.col(pivot));
      B_.col(step).setZero();
      B_(pivot, step) = 1;
      // update column with the current matrix Lmatrix_
      L_.col(step) -= L_.leftCols(step) * L_.row(pivot).head(step).transpose();
      B_.col(step) -= B_.leftCols(step) * L_.row(pivot).head(step).transpose();
      if (L_(pivot, step) <= 0) {
        info_ = 2;
        std::cout << "breaking with non positive pivot\n";
        break;
      }
      L_.col(step) /= sqrt(D(pivot));
      B_.col(step) /= sqrt(D(pivot));
      // update the diagonal and the trace
      D.array() -= L_.col(step).array().square();
      r -= r.dot(B_.col(step)) * L_.col(step);
      // compute the trace of the Schur complement
      resnorm = r.norm();
      ++step;
    }
    std::cout << "steps: " << step << " residual: " << resnorm << std::endl;
    if (D.sum() < 0)
      info_ = 2;
    else
      info_ = 0;
    // crop L, indices to their actual size
    L_.conservativeResize(dim, step);
    B_.conservativeResize(dim, step);
    indices_.conservativeResize(step);
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
    Eigen::SelfAdjointEigenSolver<Matrix> es;
    {
      Matrix K = ker.eval(P, P);
      es.compute(K);
      info_ = es.info();
      if (es.info() != Eigen::Success) return;
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
    Eigen::SelfAdjointEigenSolver<Matrix> es(C);
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
    for (Index i = 0; i < indices_.size(); ++i) U.row(i) = B_.row(indices_(i));
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
  Index info_;
  // we cap the maximum matrix size at 8GB
  const Index max_size_ = Index(1e9);
};
}  // namespace FMCA
#endif

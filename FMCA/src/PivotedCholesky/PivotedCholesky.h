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
#ifndef FMCA_PIVOTEDCHOLESKY_PIVOTEDCHOLESKY_H_
#define FMCA_PIVOTEDCHOLESKY_PIVOTEDCHOLESKY_H_

namespace FMCA {
class PivotedCholesky {
 public:
  PivotedCholesky() {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    tol_ = 0;
  }

  PivotedCholesky(const CovarianceKernel &ker, const FMCA::Matrix &P,
                  FMCA::Scalar tol = 1e-3)
      : tol_(tol) {
    L_.resize(0, 0);
    B_.resize(0, 0);
    indices_.resize(0);
    compute(ker, P, tol);
  }

  void compute(const CovarianceKernel &ker, const FMCA::Matrix &P,
               FMCA::Scalar tol = 1e-3) {
    const FMCA::Index dim = P.cols();
    const FMCA::Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    FMCA::Vector D(dim);
    FMCA::Index pivot = 0;
    FMCA::Scalar tr = 0;
    L_.resize(dim, max_cols);
    indices_.resize(max_cols);
    tol_ = tol;
    // compute the diagonal and the trace
    for (auto i = 0; i < dim; ++i) {
      const FMCA::Matrix wtf = ker.eval(P.col(i), P.col(i));
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
              << " rel tol: " << tol << " initial trace: " << tr << std::endl;
    FMCA::Index step = 0;
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

  void computeFullPiv(const CovarianceKernel &ker, const FMCA::Matrix &P,
                      FMCA::Scalar tol = 1e-3) {
    const FMCA::Index dim = P.cols();
    const FMCA::Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    tol_ = tol;
    if (max_cols < dim) {
      info_ = 3;
      return;
    }
    Eigen::SelfAdjointEigenSolver<FMCA::Matrix> es;
    {
      FMCA::Matrix K = ker.eval(P, P);
      es.compute(K);
      info_ = es.info();
      if (es.info() != Eigen::Success) return;
    }
    FMCA::Vector ev = es.eigenvalues().reverse();
    std::cout << "lambda min: " << ev.minCoeff() << " "
              << "lambda max: " << ev.maxCoeff();
    FMCA::Scalar tr = ev.sum();
    FMCA::Scalar cur_tr = 0;
    FMCA::Index step = 0;
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

  const FMCA::Matrix &matrixB() { return B_; }
  const FMCA::Matrix &matrixL() { return L_; }
  const FMCA::iVector &indices() { return indices_; }
  const FMCA::Scalar &tol() { return tol_; }
  const FMCA::Index &info() { return info_; }

 private:
  // member variables
  FMCA::Matrix L_;
  FMCA::Matrix B_;
  FMCA::iVector indices_;
  FMCA::Scalar tol_;
  FMCA::Index info_;
  // we cap the maximum matrix size at 2GB
  const FMCA::Index max_size_ = 250000000;
};
}  // namespace FMCA
#endif

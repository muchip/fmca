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
#ifndef FMCA_LOWRANKAPPROXIMATION_NYSTROMAPPROXIMATION_H_
#define FMCA_LOWRANKAPPROXIMATION_NYSTROMAPPROXIMATION_H_
#include <random>

namespace FMCA {
class NystromApproximation {
 public:
  NystromApproximation() {
    L_.resize(0, 0);
    indices_.resize(0);
  }

  NystromApproximation(const CovarianceKernel &ker, const Matrix &P,
                       const Index rank)
      : rank_(rank) {
    L_.resize(0, 0);
    indices_.resize(0);
    compute(ker, P, rank);
  }

  void compute(const CovarianceKernel &ker, const Matrix &P, const Index rank) {
    const Index dim = P.cols();
    const Index max_cols = max_size_ / dim > dim ? dim : max_size_ / dim;
    std::mt19937 mtwister_(0);
    assert(rank <= max_cols && "Nystrom rank too high");
    indices_.resize(dim);
    std::iota(indices_.begin(), indices_.end(), 0);
    std::shuffle(indices_.begin(), indices_.end(), mtwister_);
    indices_.conservativeResize(rank);
    C_.resize(P.rows(), rank);
    for (Index i = 0; i < rank; ++i) C_.col(i) = P.col(indices_(i));
    const FMCA::Matrix KCC = ker.eval(C_, C_);
    const FMCA::Matrix UTCC = KCC.llt().matrixL().transpose().solve(
        FMCA::Matrix::Identity(KCC.rows(), KCC.cols()));
    L_ = ker.eval(P, C_) * UTCC;
    return;
  }

  const Matrix &matrixL() { return L_; }
  const iVector &indices() { return indices_; }

 private:
  // member variables
  Matrix L_;
  Matrix C_;
  iVector indices_;
  Index rank_;
  // we cap the maximum matrix size at 8GB
  const Index max_size_ = Index(1e9);
};
}  // namespace FMCA
#endif

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
#ifndef FMCA_UTIL_ACTIVESETMANAGER_H_
#define FMCA_UTIL_ACTIVESETMANAGER_H_

#include "Macros.h"

namespace FMCA {
class ActiveSetManager {
 public:
  ActiveSetManager() {};
  template <typename T>
  ActiveSetManager(const T& A, const std::vector<Index>& aidcs) {
    init(A, aidcs);
    return;
  }
  template <typename T>
  void init(const T& A, const std::vector<Index>& aidcs) {
    idcs_ = std::vector<std::ptrdiff_t>(A.cols(), -1);
    U_.conservativeResize(A.rows(), aidcs.size());
    for (Index i = 0; i < aidcs.size(); ++i) {
      U_.col(i) = A.col(aidcs[i]);
      idcs_[aidcs[i]] = i;
    }
    Eigen::JacobiSVD<Matrix> svd(U_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U_ = svd.matrixU();
    sigma_ = svd.singularValues();
    sactive_ = sigma_;
    V_ = svd.matrixV();
    return;
  }

  static void QRupdate(Matrix* Q, Matrix* R, const Vector& vec) {
    Vector q = vec;
    Vector r(Q->cols());
    r.setZero();
    for (Index i = 0; i < 1; ++i) {
      const Vector c = Q->transpose() * q;
      q = q - (*Q) * c;
      r += c;
    }
    const Scalar rho = q.norm();
    if (rho > 100 * FMCA_ZERO_TOLERANCE) {
      Q->conservativeResize(Q->rows(), Q->cols() + 1);
      Q->rightCols(1) = (1. / rho) * q;
      R->conservativeResize(R->rows() + 1, R->cols() + 1);
      R->bottomRows(1).setZero();
      R->rightCols(1).topRows(r.size()) = r;
      (*R)(R->rows() - 1, R->cols() - 1) = rho;
    } else {
      R->conservativeResize(R->rows(), R->cols() + 1);
      R->rightCols(1) = r;
    }
    return;
  }

  template <typename T>
  void update(const T& A, const std::vector<Index>& aidcs) {
    FMCA::Index nnew = 0;
    for (const auto& it : aidcs) {
      if (idcs_[it] == -1) {
        Matrix S = sigma_.asDiagonal();
        QRupdate(&U_, &S, A.col(it));
        V_.conservativeResize(V_.rows() + 1, V_.cols() + 1);
        V_.rightCols(1).setZero();
        V_.bottomRows(1).setZero();
        V_(V_.rows() - 1, V_.cols() - 1) = 1;
        idcs_[it] = V_.cols() - 1;
        Eigen::JacobiSVD<Matrix> svd(S,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
        U_ = U_ * svd.matrixU();
        V_ = V_ * svd.matrixV();
        sigma_ = svd.singularValues();
        ++nnew;
      }
    }
    std::cout << "new indices: " << nnew << "/" << U_.cols() << "/" << V_.cols()
              << std::endl;
    return;
  }
  const Matrix& matrixU() const { return U_; }
  const Vector& sigma() const { return sigma_; }
  const Matrix& matrixV() const { return V_; }
  const Vector& sactive() const { return sactive_; }

  Vector activeS(std::vector<Index>& aidcs) const {
    Vector retval(aidcs.size());
    for (Index i = 0; i < aidcs.size(); ++i)
      retval(i) = sigma_(idcs_[aidcs[i]]);
    return retval;
  }

  Matrix activeV(std::vector<Index>& aidcs) const {
    Matrix retval(aidcs.size(), V_.rows());
    for (Index i = 0; i < aidcs.size(); ++i)
      retval.row(i) = V_.row(idcs_[aidcs[i]]);
    return retval;
  }

  Matrix activeVSinv(std::vector<Index>& aidcs) {
    Matrix retval(aidcs.size(), V_.rows());
    for (Index i = 0; i < aidcs.size(); ++i)
      retval.row(i) = V_.row(idcs_[aidcs[i]]);
    Eigen::JacobiSVD<Matrix> svd(retval * sigma_.asDiagonal(),
                                 Eigen::ComputeThinU | Eigen::ComputeThinV);

    sactive_ = svd.singularValues();
    const Vector s_inv = (sactive_.array() > 100 * FMCA_ZERO_TOLERANCE)
                             .select(sactive_.array().inverse(), 0.0);
    // satisfies VA * S^2 * VA^T * VSinv * VSinv^T = I
    return svd.matrixU() * s_inv.asDiagonal();
  }

  const std::vector<std::ptrdiff_t>& indices() const { return idcs_; }

 private:
  Matrix U_;
  Vector sigma_;
  Vector sactive_;
  Matrix V_;
  std::vector<std::ptrdiff_t> idcs_;
};
}  // namespace FMCA
#endif

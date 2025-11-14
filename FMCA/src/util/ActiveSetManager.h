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
    Q_.conservativeResize(A.rows(), aidcs.size());
    for (Index i = 0; i < aidcs.size(); ++i) {
      Q_.col(i) = A.col(aidcs[i]);
      idcs_[aidcs[i]] = i;
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Q_);
    Q_ = qr.householderQ() * Matrix::Identity(Q_.rows(), Q_.cols());
    R_ = qr.matrixQR().topRows(Q_.cols()).triangularView<Eigen::Upper>();
    Eigen::JacobiSVD<Matrix> svd(R_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U_ = svd.matrixU();
    S_ = svd.singularValues().asDiagonal();
    V_ = svd.matrixV();
    return;
  }

  static void QRupdate(Matrix* Q, Matrix* R, const Vector& vec) {
    Vector q = vec;
    Vector r(Q->cols());
    r.setZero();
    for (Index i = 0; i < 2; ++i) {
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
    for (const auto& it : aidcs) {
      if (idcs_[it] == -1) {
        // update QR
        QRupdate(&Q_, &R_, A.col(it));
        idcs_[it] = R_.cols() - 1;
        // update SVD
        if (R_.rows() > U_.rows()) {
          U_.conservativeResize(U_.rows() + 1, U_.cols());
          U_.bottomRows(1).setZero();
        }
        QRupdate(&U_, &S_, R_.rightCols(1));
        V_.conservativeResize(V_.rows() + 1, V_.cols() + 1);
        V_.rightCols(1).setZero();
        V_.bottomRows(1).setZero();
        V_(V_.rows() - 1, V_.cols() - 1) = 1;
        Eigen::JacobiSVD<Matrix> svd(S_,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
        U_ = U_ * svd.matrixU();
        V_ = V_ * svd.matrixV();
        S_ = svd.singularValues().asDiagonal();
      }
    }
  }
  const Matrix& matrixQ() const { return Q_; }
  const Matrix& matrixR() const { return R_; }
  const Matrix& matrixU() const { return U_; }
  const Matrix& matrixS() const { return S_; }
  const Matrix& matrixV() const { return V_; }

  Vector activeS(std::vector<Index>& aidcs) const {
    Vector retval(aidcs.size());
    for (Index i = 0; i < aidcs.size(); ++i)
      retval(i) = S_(idcs_[aidcs[i]], idcs_[aidcs[i]]);
    return retval;
  }

  Matrix activeV(std::vector<Index>& aidcs) const {
    Matrix retval(aidcs.size(), V_.rows());
    for (Index i = 0; i < aidcs.size(); ++i)
      retval.row(i) = V_.row(idcs_[aidcs[i]]);
    return retval;
  }

  Matrix activeVSinv(std::vector<Index>& aidcs) const {
    Matrix retval(aidcs.size(), V_.rows());
    for (Index i = 0; i < aidcs.size(); ++i)
      retval.row(i) = V_.row(idcs_[aidcs[i]]);
    Eigen::JacobiSVD<Matrix> svd(retval * S_,
                                 Eigen::ComputeThinU | Eigen::ComputeThinV);

    const Vector s = svd.singularValues();
    const Vector s_inv = (s.array() > 100 * FMCA_ZERO_TOLERANCE)
                             .select(s.array().inverse(), 0.0);
    // satisfies VA * S^2 * VA^T * VSinv * VSinv^T = I
    return svd.matrixU() * s_inv.asDiagonal();
  }

  const std::vector<std::ptrdiff_t>& indices() const { return idcs_; }

 private:
  Matrix Q_;
  Matrix R_;
  Matrix U_;
  Matrix S_;
  Matrix V_;
  std::vector<std::ptrdiff_t> idcs_;
};
}  // namespace FMCA
#endif

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
    if (U_.cols()) {
      JacobiSVD svd(U_, ComputeThinUV);
      U_ = svd.matrixU();
      sigma_ = svd.singularValues();
      sactive_ = sigma_;
      V_ = svd.matrixV();
    }
    return;
  }

  static void QRupdate(Matrix* Q, Matrix* R, const Matrix& C) {
    Matrix QC = C;
    Matrix RC(Q->cols(), C.cols());
    RC.setZero();
    // orthogonalize C wrt Q
    for (Index i = 0; i < 2; ++i) {
      const Matrix CC = Q->transpose() * QC;
      QC = QC - (*Q) * CC;
      RC += CC;
    }
    HouseholderQR qr(QC);
    QC = qr.householderQ() * Matrix::Identity(QC.rows(), QC.cols());
    const Matrix Rrem =
        qr.matrixQR().topRows(QC.cols()).triangularView<Upper>();

    const Scalar rho = Rrem.norm();
    if (rho > 100 * FMCA_ZERO_TOLERANCE) {
      Q->conservativeResize(Q->rows(), Q->cols() + QC.cols());
      Q->rightCols(QC.cols()) = QC;
      R->conservativeResize(R->rows() + Rrem.rows(), R->cols() + RC.cols());
      R->bottomRows(Rrem.rows()).setZero();
      R->rightCols(RC.cols()).topRows(RC.rows()) = RC;
      R->rightCols(RC.cols()).bottomRows(Rrem.rows()) = Rrem;
    } else {
      R->conservativeResize(R->rows(), R->cols() + RC.cols());
      R->rightCols(RC.cols()) = RC;
    }
    return;
  }

  template <typename T>
  void update(const T& A, const std::vector<Index>& aidcs) {
    FMCA::Index nnew = 0;
    FMCA::Index nold = 0;
    for (const auto& it : aidcs)
      if (idcs_[it] == -1)
        ++nnew;
      else
        ++nold;
    Matrix C(A.rows(), nnew);
    if (nnew) {
      // fill C
      Index i = 0;
      for (const auto& it : aidcs)
        if (idcs_[it] == -1) {
          C.col(i) = A.col(it);
          idcs_[it] = V_.cols() + i;
          ++i;
        }
      Matrix S = sigma_.asDiagonal();
      QRupdate(&U_, &S, C);
      V_.conservativeResize(V_.rows() + nnew, V_.cols() + nnew);
      V_.rightCols(nnew).setZero();
      V_.bottomRows(nnew).setZero();
      V_.bottomRows(nnew).rightCols(nnew).setIdentity();

      JacobiSVD svd(S, ComputeThinUV);
      // std::cout << "[ActiveSetManager] SVD recomputed, added "
      //     << nnew << " columns" << std::endl;

      U_ = U_ * svd.matrixU();
      V_ = V_ * svd.matrixV();
      sigma_ = svd.singularValues();
    }
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
    JacobiSVD svd(retval * sigma_.asDiagonal(), ComputeThinUV);

    sactive_ = svd.singularValues();
    const Scalar trace = sactive_.sum();
    const Vector s_inv = (sactive_.array() > 1e-4 * trace)
                             .select(sactive_.array().inverse(), 1e4 / trace);
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

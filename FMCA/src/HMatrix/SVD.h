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
#ifndef FMCA_HMATRIX_SVD_H_
#define FMCA_HMATRIX_SVD_H_

namespace FMCA {

template <typename MatrixEvaluator, typename RowCType, typename ColCType>
void SVD(const MatrixEvaluator &mat_eval, const RowCType &rc,
         const ColCType &cc, Matrix *L, Matrix *R, const Scalar tol) {
  Eigen::BDCSVD<Matrix> svd;
  Matrix A(rc.block_size(), cc.block_size());
  for (Index j = 0; j < A.cols(); ++j)
    for (Index i = 0; i < A.rows(); ++i)
      A(i, j) = mat_eval(rc.indices()[i], cc.indices()[j]);
  svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Scalar fnorm2 = svd.singularValues().squaredNorm();
  Index rank = 0;
  Scalar cur_fnorm2;
  while (fnorm2 - cur_fnorm2 > tol * tol * fnorm2) {
    cur_fnorm2 += svd.singularValues()[rank] * svd.singularValues()[rank];
    ++rank;
  }
  *L = svd.matrixU().leftCols(rank);
  *R = svd.matrixV().leftCols(rank) *
       svd.singularValues().head(rank).asDiagonal();
  return;
}

}  // namespace FMCA
#endif

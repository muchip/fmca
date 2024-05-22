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
#ifndef FMCA_HMATRIX_ACA_H_
#define FMCA_HMATRIX_ACA_H_

namespace FMCA {

template <typename MatrixEvaluator, typename RowCType, typename ColCType>
void ACA(const MatrixEvaluator &mat_eval, const RowCType &rc,
         const ColCType &cc, Matrix *L, Matrix *R, const Scalar tol) {
  const Index allocBSize = 100;
  Scalar max = 0;
  Scalar normS = 0;
  Index actBSize = 0;
  Index step = 0;
  Index pivl = 0;
  Index pivc = 0;
  // allocate memory for L and R
  actBSize = allocBSize;
  L->resize(rc.block_size(), actBSize);
  R->resize(cc.block_size(), actBSize);
  // start Gaussian elimination of A
  while (step < rc.block_size() && step < cc.block_size()) {
    // guarantee that there is sufficient memory. otherwise reallocate
    if (actBSize - 1 <= step) {
      actBSize += allocBSize;
      L->conservativeResize(L->rows(), actBSize);
      R->conservativeResize(R->rows(), actBSize);
    }
    // extract new row;
    for (Index i = 0; i < R->rows(); ++i)
      (*R)(i, step) = mat_eval(rc.indices()[pivl], cc.indices()[i]);

    // update new row;
    if (step > 0)
      (*R).col(step) -= (*R).block(0, 0, R->rows(), step) *
                        (*L).row(pivl).head(step).transpose();
    // get new column pivot
    max = (*R).col(step).cwiseAbs().maxCoeff(&pivc);
    // error check to ensure that code does not crash due to small pivot
    if (max < FMCA_ZERO_TOLERANCE && step > 0) {
      L->conservativeResize(L->rows(), step);
      R->conservativeResize(R->rows(), step);
      return;
    }
    // scale row by the pivot element
    (*R).col(step) *= 1. / (*R)(pivc, step);
    // extract new column
    // extract new row;
    for (Index i = 0; i < L->rows(); ++i)
      (*L)(i, step) = mat_eval(rc.indices()[i], cc.indices()[pivc]);
    if (step > 0)
      (*L).col(step) -= (*L).block(0, 0, L->rows(), step) *
                        (*R).row(pivc).head(step).transpose();
    // determine new pivot line
    max = 0;
    Index pivlold = pivl;
    for (auto i = 0; i < L->rows(); ++i)
      if (max < std::abs((*L)(i, step)) && i != pivlold) {
        max = std::abs((*L)(i, step));
        pivl = i;
      }
    // check the fancy error criterion according to
    // [Bebendorf,Rjasanow: Adaptive Low Approximation of Collocation Matrices]
    Scalar sum = 0;
    for (auto i = 0; i < step; ++i) {
      Scalar innR = (*R).col(i).dot((*R).col(step));
      Scalar innL = (*L).col(i).dot((*L).col(step));
      sum += innL * innR;
    }
    Scalar normR = (*R).col(step).norm();
    Scalar normL = (*L).col(step).norm();
    normS += 2 * sum + normL * normL * normR * normR;
    ++step;
    if (normL * normR <= tol * std::sqrt(normS)) break;
  }
  L->conservativeResize(L->rows(), step);
  R->conservativeResize(R->rows(), step);
  return;
}

}  // namespace FMCA
#endif

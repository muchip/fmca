// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_H2MATRIX_MATRIXVECTORPRODUCTIMPL_H_
#define FMCA_H2MATRIX_MATRIXVECTORPRODUCTIMPL_H_

template <typename Derived, typename otherDerived>
typename Derived::eigenMatrix
matrix_vector_product_impl(const H2Matrix<Derived> &H2,
                           const Eigen::MatrixBase<otherDerived> &rhs) {
  typename Derived::eigenMatrix retval(mat.rows(), mat.cols());
  retval.setZero();
  return retval;
}

#endif

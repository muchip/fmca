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
typename Derived::eigenMatrix matrix_vector_product_impl(
    Derived &H2, const Eigen::MatrixBase<otherDerived> &rhs) {
  typedef typename Derived::eigenMatrix eigenMatrix;
  eigenMatrix lhs(rhs.rows(), rhs.cols());
  lhs.setZero();
  std::vector<eigenMatrix> trhs = forward_transform_impl(H2, rhs);
  std::vector<eigenMatrix> tlhs = trhs;
  for (auto &&it : tlhs) it.setZero();
  for (auto &it : H2) {
    // there is something to multiply
    if (!it.sons().size()) {
      if (it.is_low_rank())
        tlhs[it.rcluster()->block_id()] +=
            it.matrixS() * trhs[it.ccluster()->block_id()];
      else
        lhs.middleRows((it.rcluster())->indices_begin(), it.matrixS().rows()) +=
            it.matrixS() * rhs.middleRows((it.ccluster())->indices_begin(),
                                          it.matrixS().cols());
    }
  }
  backward_transform_recursion(*(H2.rcluster()), &lhs, tlhs);
  return lhs;
}

#endif

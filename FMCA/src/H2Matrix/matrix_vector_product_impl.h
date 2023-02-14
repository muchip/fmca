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
#ifndef FMCA_H2MATRIX_MATRIXVECTORPRODUCTIMPL_H_
#define FMCA_H2MATRIX_MATRIXVECTORPRODUCTIMPL_H_

namespace FMCA {
namespace internal {
template <typename Derived>
Matrix matrix_vector_product_impl(const Derived &H2, const Matrix &rhs) {
  Matrix lhs(H2.rows(), rhs.cols());
  lhs.setZero();
  std::vector<Matrix> trhs(H2.ncclusters());
  forward_transform_recursion(*(H2.ccluster()), &trhs, rhs);
  std::vector<Matrix> tlhs(H2.nrclusters());
  for (const auto &it : *(H2.rcluster())) {
    if (it.nSons())
      tlhs[it.block_id()].resize(it.Es()[0].rows(), rhs.cols());
    else
      tlhs[it.block_id()].resize(it.V().rows(), rhs.cols());
    tlhs[it.block_id()].setZero();
  }
  for (const auto &it : H2) {
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
}  // namespace internal
}  // namespace FMCA
#endif

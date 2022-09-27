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
template <typename Derived, typename otherDerived>
Matrix matrix_vector_product_impl(const Derived &H2,
                                  const Eigen::MatrixBase<otherDerived> &rhs) {
  Matrix lhs(rhs.rows(), rhs.cols());
  lhs.setZero();
  std::vector<Matrix> trhs = forward_transform_impl(H2, rhs);
  std::vector<Matrix> tlhs = trhs;
  for (auto &&it : tlhs) it.setZero();
  for (auto it = H2.cbegin(); it != H2.cend(); ++it) {
    // there is something to multiply
    if (!it->sons().size()) {
      if (it->is_low_rank())
        tlhs[it->rcluster()->block_id()] +=
            it->matrixS() * trhs[it->ccluster()->block_id()];
      else
        lhs.middleRows((it->rcluster())->indices_begin(),
                       it->matrixS().rows()) +=
            it->matrixS() * rhs.middleRows((it->ccluster())->indices_begin(),
                                           it->matrixS().cols());
    }
  }
  backward_transform_recursion(*(H2.rcluster()), &lhs, tlhs);
  return lhs;
}
}  // namespace internal
}  // namespace FMCA
#endif

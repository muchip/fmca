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
#ifndef FMCA_H2MATRIX_H2FORWRDTRANSFORM_IMPL_H_
#define FMCA_H2MATRIX_H2FORWRDTRANSFORM_IMPL_H_

namespace FMCA {
namespace internal {
/**
 *  \ingroup H2Matrix
 *  \brief implements the forward transform for the matrix times vector product
 */
template <typename Derived1, typename Derived2, typename Derived3>
void forward_transform_recursion(const H2ClusterTree<Derived1> &CT,
                                 Derived2 *tvec,
                                 const Eigen::MatrixBase<Derived3> &vec) {
  if (CT.nSons()) {
    (*tvec)[CT.block_id()].resize(CT.Es()[0].rows(), vec.cols());
    (*tvec)[CT.block_id()].setZero();
    for (auto i = 0; i < CT.nSons(); ++i) {
      forward_transform_recursion(CT.sons(i), tvec, vec);
      (*tvec)[CT.block_id()] += CT.Es()[i] * (*tvec)[CT.sons(i).block_id()];
    }
  } else {
    (*tvec)[CT.block_id()] =
        CT.node().V_ * vec.middleRows(CT.indices_begin(), CT.indices().size());
  }
}

template <typename Derived, typename otherDerived>
std::vector<Matrix> forward_transform_impl(
    const Derived &mat, const Eigen::MatrixBase<otherDerived> &vec) {
  std::vector<Matrix> retval(mat.nclusters());
  forward_transform_recursion(*(mat.ccluster()), &retval, vec);
  return retval;
};
}  // namespace internal
}  // namespace FMCA
#endif

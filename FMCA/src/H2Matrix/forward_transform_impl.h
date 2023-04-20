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
template <typename Derived, typename T = Matrix>
void forward_transform_recursion(const H2ClusterTree<Derived> &ct,
                                 std::vector<T> *tvec, const T &vec) {
  if (ct.nSons()) {
    (*tvec)[ct.block_id()].resize(ct.Es()[0].rows(), vec.cols());
    (*tvec)[ct.block_id()].setZero();
    for (auto i = 0; i < ct.nSons(); ++i) {
      forward_transform_recursion(ct.sons(i), tvec, vec);
      (*tvec)[ct.block_id()] += ct.Es()[i] * (*tvec)[ct.sons(i).block_id()];
    }
  } else {
    (*tvec)[ct.block_id()] =
        ct.node().V_ * vec.middleRows(ct.indices_begin(), ct.indices().size());
  }
}

template <typename Derived, typename T = Matrix>
std::vector<T> forward_transform_impl(const H2MatrixBase<Derived> &mat,
                                      const T &vec) {
  std::vector<T> retval(mat.ncclusters());
  forward_transform_recursion(*(mat.ccluster()), &retval, vec);
  return retval;
};
}  // namespace internal
}  // namespace FMCA
#endif

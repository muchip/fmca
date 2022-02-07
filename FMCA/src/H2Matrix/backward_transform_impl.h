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
#ifndef FMCA_H2MATRIX_H2BACKWARDTRANSFORM_IMPL_H_
#define FMCA_H2MATRIX_H2BACKWARDTRANSFORM_IMPL_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief implements the forward transform for the matrix times vector product
 */
template <typename Derived, typename otherDerived>
void backward_transform_recursion(const H2ClusterTree &CT, Derived *tvec,
                                  otherDerived &vec) {
  if (CT.nSons()) {
    for (auto i = 0; i < CT.nSons(); ++i) {
      vec[CT.sons(i).block_id()] += CT.Es()[i].transpose() * vec[CT.block_id()];
      backward_transform_recursion(CT.sons(i), tvec, vec);
    }
  } else {
    tvec->middleRows(CT.indices_begin(), CT.V().cols()) +=
        CT.V().transpose() * vec[CT.block_id()];
  }
}

template <typename Derived>
typename Derived::eigenMatrix
backward_transform_impl(const Derived &mat,
                        std::vector<typename Derived::eigenMatrix> &vec) {
  typename Derived::eigenMatrix retval(mat.rcluster()->indices().size(),
                                       vec[0].cols());
  retval.setZero();
  backward_transform_recursion(*(mat.rcluster()), &retval, vec);
  return retval;
};
} // namespace FMCA
#endif

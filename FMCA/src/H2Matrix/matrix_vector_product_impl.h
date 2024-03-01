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
  std::vector<std::vector<const Derived *>> scheduler;
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
  scheduler.resize(H2.ncclusters());
  for (const auto &it : H2)
    if (!it.nSons())
      scheduler[it.ccluster()->block_id()].push_back(std::addressof(it));
  for (const auto &it2 : scheduler) {
#pragma omp parallel for schedule(dynamic)
    for (Index k = 0; k < it2.size(); ++k) {
      const Derived &it = *(it2[k]);
      const Index i = it.rcluster()->block_id();
      const Index j = it.ccluster()->block_id();
      const Index ii = (it.rcluster())->indices_begin();
      const Index jj = (it.ccluster())->indices_begin();
      if (it.is_low_rank()) {
        tlhs[i] += it.matrixS() * trhs[j];
      } else {
        lhs.middleRows(ii, it.matrixS().rows()) +=
            it.matrixS() * rhs.middleRows(jj, it.matrixS().cols());
      }
    }
  }
#if 0
  Index pos = 0;
#pragma omp parallel shared(pos)
  {
    Index i = 0;
    Index prev_i = 0;
    typename Derived::const_iterator it = H2.begin();
#pragma omp atomic capture
    i = pos++;
    while (it != H2.end()) {
      Index dist = i - prev_i;
      while (dist > 0 && it != H2.end()) {
        --dist;
        ++it;
      }
      if (it == H2.end()) break;
      if (!(it->nSons())) {
        if (it->is_low_rank())
          tlhs[it->rcluster()->block_id()] +=
              it->matrixS() * trhs[it->ccluster()->block_id()];
        else
          lhs.middleRows((it->rcluster())->indices_begin(),
                         it->matrixS().rows()) +=
              it->matrixS() * rhs.middleRows((it->ccluster())->indices_begin(),
                                             it->matrixS().cols());
      }
      prev_i = i;
#pragma omp atomic capture
      i = pos++;
    }
  }
  for (const auto &it : H2) {
    // there is something to multiply
    if (!it.nSons()) {
      if (it.is_low_rank())
        tlhs[it.rcluster()->block_id()] +=
            it.matrixS() * trhs[it.ccluster()->block_id()];
      else
        lhs.middleRows((it.rcluster())->indices_begin(), it.matrixS().rows()) +=
            it.matrixS() * rhs.middleRows((it.ccluster())->indices_begin(),
                                          it.matrixS().cols());
    }
  }
#endif
  backward_transform_recursion(*(H2.rcluster()), &lhs, tlhs);
  return lhs;
}
}  // namespace internal
}  // namespace FMCA
#endif

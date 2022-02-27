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
#ifndef FMCA_H2MATRIX_COMPUTECLUSTERBASESIMPL_H_
#define FMCA_H2MATRIX_COMPUTECLUSTERBASESIMPL_H_

namespace FMCA {
namespace internal {

/**
 * \ingroup internal
 * \brief Removed the type checker here as this is in any case an internal
 *        method. This way, we can reuse it for the H2SampletTree. At the
 *        moment there is no more elegant solution to avoid the diamond
 *        when inheriting from H2ClusterTreeBase and SampletTreeBase
 **/
struct compute_cluster_bases_impl {
  compute_cluster_bases_impl() {}

  template <typename Derived, typename Moments>
  static void compute(TreeBase<Derived> &CT, const Moments &mom) {
    using eigenMatrix = typename Derived::eigenMatrix;
    Derived &H2T = CT.derived();
    H2T.V().resize(0, 0);
    H2T.Es().clear();

    if (H2T.nSons()) {
      for (auto i = 0; i < H2T.nSons(); ++i)
        compute(H2T.sons(i), mom);
      // compute transfer matrices
      const eigenMatrix &Xi = mom.interp().Xi();
      for (auto i = 0; i < H2T.nSons(); ++i) {
        eigenMatrix E(Xi.cols(), Xi.cols());
        for (auto j = 0; j < E.cols(); ++j)
          E.col(j) = mom.interp().evalPolynomials(
              (Xi.col(j).array() * H2T.sons(i).bb().col(2).array() /
                   H2T.bb().col(2).array() +
               (H2T.sons(i).bb().col(0).array() - H2T.bb().col(0).array()) /
                   H2T.bb().col(2).array())
                  .matrix());
        E = E * mom.interp().invV().transpose();
        H2T.Es().emplace_back(std::move(E));
      }
    } else
      // compute leaf bases
      H2T.V() = mom.moment_matrix(H2T);

    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename Moments>
  static int check_transfer_matrices(TreeBase<Derived> &CT,
                                     const Moments &mom) {
    using eigenMatrix = typename Derived::eigenMatrix;
    Derived &H2T = CT.derived();
    if (H2T.nSons()) {
      // check transfer matrices of sons first
      for (auto i = 0; i < H2T.nSons(); ++i)
        check_transfer_matrices(H2T.sons(i), mom);
      // now check own transfer matrix using sons transfer matrices
      eigenMatrix V = mom.moment_matrix(H2T);
      for (auto i = 0; i < H2T.nSons(); ++i) {
        H2T.V().conservativeResize(H2T.sons(i).V().rows(),
                                   H2T.V().cols() + H2T.sons(i).V().cols());
        H2T.V().rightCols(H2T.sons(i).V().cols()) =
            H2T.Es()[i] * H2T.sons(i).V();
      }
      FloatType nrm = (V - H2T.V()).norm() / V.norm();
      eigen_assert(nrm < 1e-14 && "the H2 cluster basis is faulty");
    }
    return 0;
  }
};

} // namespace internal

} // namespace FMCA

#endif

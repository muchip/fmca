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
#ifndef FMCA_H2MATRIX_COMPUTEVARIABLEORDERCLUSTERBASESIMPL_H_
#define FMCA_H2MATRIX_COMPUTEVARIABLEORDERCLUSTERBASESIMPL_H_

namespace FMCA {
namespace internal {

/**
 * \ingroup internal
 * \brief Removed the type checker here as this is in any case an internal
 *        method. This way, we can reuse it for the H2SampletTree. At the
 *        moment there is no more elegant solution to avoid the diamond
 *        when inheriting from H2ClusterTreeBase and SampletTreeBase
 **/
struct compute_variable_order_cluster_bases_impl {
  compute_variable_order_cluster_bases_impl() {}

  template <typename Derived, typename Moments>
  static void compute(TreeBase<Derived> &CT, const Moments &mom) {
    Derived &H2T = CT.derived();
    H2T.V().resize(0, 0);
    H2T.Es().clear();

    if (H2T.nSons()) {
      for (auto i = 0; i < H2T.nSons(); ++i) compute(H2T.sons(i), mom);
      // compute transfer matrices
      for (auto i = 0; i < H2T.nSons(); ++i) {
        const Matrix &Xi = mom.interp(H2T.sons(i).level()).Xi();
        Matrix E(Xi.cols(), Xi.cols());
        for (auto j = 0; j < E.cols(); ++j)
          E.col(j) = mom.interp(H2T.level())
                         .evalPolynomials((Xi.col(j).array() *
                                               H2T.sons(i).bb().col(2).array() /
                                               H2T.bb().col(2).array() +
                                           (H2T.sons(i).bb().col(0).array() -
                                            H2T.bb().col(0).array()) /
                                               H2T.bb().col(2).array())
                                              .matrix());
        E = E * mom.interp(H2T.sons(i).level()).invV().transpose();
        H2T.Es().emplace_back(std::move(E));
      }
    } else
      // compute leaf bases
      H2T.V() = mom.moment_matrix(H2T, H2T.level());

    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived, typename Moments>
  static int check_variable_order_transfer_matrices(TreeBase<Derived> &CT,
                                                    const Moments &mom) {
    Derived &H2T = CT.derived();
    if (H2T.nSons()) {
      // check transfer matrices of sons first
      for (auto i = 0; i < H2T.nSons(); ++i)
        check_transfer_matrices(H2T.sons(i), mom);
      // now check own transfer matrix using sons transfer matrices
      Matrix V = mom.moment_matrix(H2T, H2T.level());
      for (auto i = 0; i < H2T.nSons(); ++i) {
        H2T.V().conservativeResize(H2T.sons(i).V().rows(),
                                   H2T.V().cols() + H2T.sons(i).V().cols());
        H2T.V().rightCols(H2T.sons(i).V().cols()) =
            H2T.Es()[i] * H2T.sons(i).V();
      }
      Scalar nrm = (V - H2T.V()).norm() / V.norm();
      if (nrm >= 1e-13) std::cout << nrm << std::endl;
      eigen_assert(nrm < 1e-13 && "the H2 cluster basis is faulty");
    }
    return 0;
  }
};

}  // namespace internal

}  // namespace FMCA

#endif

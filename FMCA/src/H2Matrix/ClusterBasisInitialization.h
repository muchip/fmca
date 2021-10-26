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

//#define CHECK_TRANSFER_MATRICES_
namespace FMCA {
namespace internal {

/**
 * \ingroup internal
 * \brief Removed the type checker here as this is in any case an internal
 *        method. This way, we can reuse it for the H2SampletTree. At the
 *        moment there is no more elegant solution to avoid the diamond
 *        when inheriting from H2ClusterTreeBase and SampletTreeBase
 **/
template <typename Derived, typename eigenMatrix>
void compute_cluster_bases_impl(TreeBase<Derived> &CT, const eigenMatrix &P) {
  Derived &H2T = CT.derived();
  H2T.V().resize(0, 0);
  H2T.Es().clear();

  if (H2T.nSons()) {
    for (auto i = 0; i < H2T.nSons(); ++i) {
      // give also the son access to the interpolation routines ...
      H2T.sons(i).node().interp_ = H2T.node().interp_;
      compute_cluster_bases_impl(H2T.sons(i), P);
    }
    // compute transfer matrices
    const eigenMatrix &Xi = H2T.Xi();
    for (auto i = 0; i < H2T.nSons(); ++i) {
      eigenMatrix E(Xi.cols(), Xi.cols());
      for (auto j = 0; j < E.cols(); ++j)
        E.col(j) = H2T.node().interp_->evalLagrangePolynomials(
            (Xi.col(j).array() * H2T.sons(i).bb().col(2).array() /
                 H2T.bb().col(2).array() +
             (H2T.sons(i).bb().col(0).array() - H2T.bb().col(0).array()) /
                 H2T.bb().col(2).array())
                .matrix());
      H2T.Es().emplace_back(std::move(E));
    }
  } else {
    // compute leaf bases
    H2T.V().resize(H2T.Xi().cols(), H2T.indices().size());
    for (auto i = 0; i < H2T.indices().size(); ++i)
      H2T.V().col(i) = H2T.node().interp_->evalLagrangePolynomials(
          ((P.col(H2T.indices()[i]) - H2T.bb().col(0)).array() /
           H2T.bb().col(2).array())
              .matrix());
  }
#ifdef CHECK_TRANSFER_MATRICES_
  if (H2T.node().E_.size()) {
    eigenMatrix V;
    H2T.node().V_.resize(H2T.node().interp_->Xi().cols(), H2T.indices().size());
    for (auto i = 0; i < H2T.indices().size(); ++i)
      H2T.node().V_.col(i) = H2T.node().interp_->evalLagrangePolynomials(
          ((P.col(H2T.indices()[i]) - H2T.bb().col(0)).array() /
           H2T.bb().col(2).array())
              .matrix());

    for (auto i = 0; i < H2T.nSons(); ++i) {
      V.conservativeResize(H2T.sons(i).node().V_.rows(),
                           V.cols() + H2T.sons(i).node().V_.cols());
      V.rightCols(H2T.sons(i).node().V_.cols()) =
          H2T.node().E_[i] * H2T.sons(i).node().V_;
    }
    FloatType nrm = (V - H2T.node().V_).norm() / H2T.node().V_.norm();
    eigen_assert(nrm < 1e-14 && "the H2 cluster basis is faulty");
  }
#endif
  return;
}

}  // namespace internal

}  // namespace FMCA

#endif

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

#define CHECK_TRANSFER_MATRICES_
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
void compute_cluster_bases_impl(Derived &CT, const eigenMatrix &P) {
  CT.V().resize(0, 0);
  CT.Es().clear();

  if (CT.nSons()) {
    for (auto i = 0; i < CT.nSons(); ++i) {
      // give also the son access to the interpolation routines ...
      CT.sons(i).node().interp_ = CT.node().interp_;
      compute_cluster_bases_impl(CT.sons(i), P);
    }
    // compute transfer matrices
    const eigenMatrix &Xi = CT.Xi();
    for (auto i = 0; i < CT.nSons(); ++i) {
      eigenMatrix E(Xi.cols(), Xi.cols());
      for (auto j = 0; j < E.cols(); ++j)
        E.col(j) = CT.node().interp_->evalLagrangePolynomials(
            (Xi.col(j).array() * CT.sons(i).bb().col(2).array() /
                 CT.bb().col(2).array() +
             (CT.sons(i).bb().col(0).array() - CT.bb().col(0).array()) /
                 CT.bb().col(2).array())
                .matrix());
      CT.Es().emplace_back(std::move(E));
    }
  } else {
    // compute leaf bases
    CT.V().resize(CT.Xi().cols(), CT.indices().size());
    for (auto i = 0; i < CT.indices().size(); ++i)
      CT.V().col(i) = CT.node().interp_->evalLagrangePolynomials(
          ((P.col(CT.indices()[i]) - CT.bb().col(0)).array() /
           CT.bb().col(2).array())
              .matrix());
  }
#ifdef CHECK_TRANSFER_MATRICES_
  if (CT.node().E_.size()) {
    eigenMatrix V;
    CT.node().V_.resize(CT.node().interp_->Xi().cols(), CT.indices().size());
    for (auto i = 0; i < CT.indices().size(); ++i)
      CT.node().V_.col(i) = CT.node().interp_->evalLagrangePolynomials(
          ((P.col(CT.indices()[i]) - CT.bb().col(0)).array() /
           CT.bb().col(2).array())
              .matrix());

    for (auto i = 0; i < CT.nSons(); ++i) {
      V.conservativeResize(CT.sons(i).node().V_.rows(),
                           V.cols() + CT.sons(i).node().V_.cols());
      V.rightCols(CT.sons(i).node().V_.cols()) =
          CT.node().E_[i] * CT.sons(i).node().V_;
    }
    FloatType nrm = (V - CT.node().V_).norm() / CT.node().V_.norm();
    eigen_assert(nrm < 1e-14 && "the H2 cluster basis is faulty");
  }
#endif
  return;
}

} // namespace internal

} // namespace FMCA

#endif

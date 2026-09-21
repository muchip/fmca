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
#ifndef FMCA_INTERPOLATORS_MONOMIALINTERPOLATOR_H_
#define FMCA_INTERPOLATORS_MONOMIALINTERPOLATOR_H_

namespace FMCA {

/**
 *  \brief Multivariate total degree monomial interpolator
 **/
class MonomialInterpolator {
 public:
  void init(Index dim, Index deg) {
    dim_ = dim;
    deg_ = deg;
    idcs_.init(dim, deg);
    TD_xi_.resize(dim_, idcs_.index_set().size());
    V_.resize(idcs_.index_set().size(), idcs_.index_set().size());
    // determine tensor product interpolation points
    Index k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < it.size(); ++i) TD_xi_(i, k) = LejaPoints[it[i]];
      V_.row(k) = evalPolynomials(TD_xi_.col(k)).transpose();
      ++k;
    }
    invV_ = V_.inverse();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  Matrix evalPolynomials(const MatrixBase<Derived> &pt) const {
    Vector retval(idcs_.index_set().size());
    retval.setOnes();
    Index k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < dim_; ++i)
        if (it[i]) retval(k) *= std::pow(pt(i), it[i]);
      ++k;
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Index dim() const { return dim_; }
  const Matrix &Xi() const { return TD_xi_; }
  const Matrix &invV() const { return invV_; }
  const Matrix &V() const { return V_; }
  const MultiIndexSet<TotalDegree> &idcs() const { return idcs_; }

 private:
  MultiIndexSet<TotalDegree> idcs_;
  Matrix TD_xi_;
  Matrix invV_;
  Matrix V_;
  Index dim_;
  Index deg_;
};
}  // namespace FMCA
#endif

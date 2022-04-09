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
#ifndef FMCA_INTERPOLATORS_MONOMIALINTERPOLATOR_H_
#define FMCA_INTERPOLATORS_MONOMIALINTERPOLATOR_H_

namespace FMCA {

/**
 *  \brief Multivariate total degree monomial interpolator
 **/
template <typename ValueType> class MonomialInterpolator {
public:
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, 1> eigenVector;
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;

  void init(IndexType dim, IndexType deg) {
    dim_ = dim;
    deg_ = deg;
    idcs_.init(dim, deg);
    TD_xi_.resize(dim_, idcs_.index_set().size());
    V_.resize(idcs_.index_set().size(), idcs_.index_set().size());
    // determine tensor product interpolation points
    IndexType k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < it.size(); ++i)
        TD_xi_(i, k) = LejaPoints[it[i]];
      V_.row(k) = evalPolynomials(TD_xi_.col(k)).transpose();
      ++k;
    }
    invV_ = V_.inverse();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  eigenMatrix evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    eigenVector retval(idcs_.index_set().size());
    retval.setOnes();
    IndexType k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < dim_; ++i)
        if (it[i])
          retval(k) *= std::pow(pt(i), it[i]);
      ++k;
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &Xi() const { return TD_xi_; }
  const eigenMatrix &invV() const { return invV_; }
  const eigenMatrix &V() const { return V_; }
  const MultiIndexSet<TotalDegree> &idcs() const { return idcs_; }

private:
  MultiIndexSet<TotalDegree> idcs_;
  eigenMatrix TD_xi_;
  eigenMatrix invV_;
  eigenMatrix V_;
  IndexType dim_;
  IndexType deg_;
};
} // namespace FMCA
#endif

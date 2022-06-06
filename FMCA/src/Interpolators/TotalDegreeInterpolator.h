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
#ifndef FMCA_INTERPOLATORS_TOTALDEGREEINTERPOLATOR_H_
#define FMCA_INTERPOLATORS_TOTALDEGREEINTERPOLATOR_H_

namespace FMCA {

class TotalDegreeInterpolator {
public:
  /**
   *  \brief These are the corresponding weights of the Chebyshev nodes
   *         for barycentric interpolation. see [1]. Note: The scaling is wrong
   *         as the nodes are on [0,1]. However, this does not matter as
   *         the factor cancels.
   **/
  void init(Index dim, Index deg) {
    dim_ = dim;
    deg_ = deg;
    idcs_.init(dim, deg);
    TD_xi_.resize(dim_, idcs_.index_set().size());
    V_.resize(idcs_.index_set().size(), idcs_.index_set().size());
    // determine tensor product interpolation points
    Index k = 0;
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
  Matrix evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    Vector retval(idcs_.index_set().size());
    Matrix p_values = evalLegendrePolynomials(deg_, pt);
    retval.setOnes();
    Index k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < dim_; ++i)
        retval(k) *= p_values(i, it[i]);
      ++k;
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Matrix &Xi() const { return TD_xi_; }
  const Matrix &invV() const { return invV_; }
  const Matrix &V() const { return V_; }
  const Index dim() const { return dim_; }
  const Index deg() const { return deg_; }
  const MultiIndexSet<TotalDegree> &idcs() const { return idcs_; }

private:
  MultiIndexSet<TotalDegree> idcs_;
  Matrix TD_xi_;
  Matrix invV_;
  Matrix V_;
  Index dim_;
  Index deg_;
};
} // namespace FMCA
#endif

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
#ifndef FMCA_INTERPOLATORS_WEIGHTEDTOTALDEGREEINTERPOLATOR_H_
#define FMCA_INTERPOLATORS_WEIGHTEDTOTALDEGREEINTERPOLATOR_H_

namespace FMCA {

class WeightedTotalDegreeInterpolator {
public:
  void init(IndexType deg, const std::vector<Scalar> &weights) {
    dim_ = weights.size();
    deg_ = deg;
    idcs_.init(dim_, deg_, weights);
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
  Matrix evalLegendrePolynomials1D(const Eigen::MatrixBase<Derived> &pt) const {
    Matrix retval(pt.rows(), deg_ + 1);
    Vector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setZero();
    P1.setOnes();
    retval.col(0) = P1;
    for (auto i = 1; i <= deg_; ++i) {
      retval.col(i) = Scalar(2 * i - 1) / Scalar(i) *
                          (2 * pt.array() - 1) * P1.array() -
                      Scalar(i - 1) / Scalar(i) * P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  Vector evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    Vector retval(idcs_.index_set().size());
    Matrix p_values = evalLegendrePolynomials1D(pt);
    retval.setOnes();
    IndexType k = 0;
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

private:
  MultiIndexSet<WeightedTotalDegree> idcs_;
  Matrix TD_xi_;
  Matrix invV_;
  Matrix V_;
  IndexType dim_;
  IndexType deg_;
};
} // namespace FMCA
#endif

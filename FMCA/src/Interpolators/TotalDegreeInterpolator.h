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

template <typename ValueType> class TotalDegreeInterpolator {
public:
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, 1> eigenVector;
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  /**
   *  \brief These are the corresponding weights of the Chebyshev nodes
   *         for barycentric interpolation. see [1]. Note: The scaling is wrong
   *         as the nodes are on [0,1]. However, this does not matter as
   *         the factor cancels.
   **/
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
  eigenMatrix
  evalLegendrePolynomials1D(const Eigen::MatrixBase<Derived> &pt) const {
    eigenMatrix retval(pt.rows(), deg_ + 1);
    eigenVector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setZero();
    P1.setOnes();
    retval.col(0) = P1;
    for (auto i = 1; i <= deg_; ++i) {
      retval.col(i) = ValueType(2 * i - 1) / ValueType(i) *
                          (2 * pt.array() - 1) * P1.array() -
                      ValueType(i - 1) / ValueType(i) * P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  eigenMatrix evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    eigenVector retval(idcs_.index_set().size());
    eigenMatrix p_values = evalLegendrePolynomials1D(pt);
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
  const eigenMatrix &Xi() const { return TD_xi_; }
  const eigenMatrix &invV() const { return invV_; }
  const eigenMatrix &V() const { return V_; }

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

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
#ifndef FMCA_INTERPOLATORS_TOTALDEGREEINTERPOLATOR_H_
#define FMCA_INTERPOLATORS_TOTALDEGREEINTERPOLATOR_H_

#include "evalPolynomials.h"

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
#if 0
    // this is using Leja points for interpolation
    Index k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < it.size(); ++i) TD_xi_(i, k) = LejaPoints[it[i]];
      V_.row(k) = internal::evalPolynomials(idcs_, TD_xi_.col(k)).transpose();
      ++k;
    }
#else
    TD_xi_ = DeMarchiPoints(idcs_);
    for (Index i = 0; i < TD_xi_.cols(); ++i)
      V_.row(i) = internal::evalPolynomials(idcs_, TD_xi_.col(i)).transpose();
#endif
    invV_ = V_.inverse();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  const Matrix &Xi() const { return TD_xi_; }
  const Matrix &invV() const { return invV_; }
  const Matrix &V() const { return V_; }
  const Index dim() const { return dim_; }
  const Index deg() const { return deg_; }
  const MultiIndexSet<TotalDegree> &idcs() const { return idcs_; }

  template <typename Derived>
  Matrix evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    return internal::evalPolynomials(idcs_, pt);
  }

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

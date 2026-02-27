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
///
#ifndef FMCA_INTERPOLATORS_TENSORPRODUCTINTERPOLATOR_H_
#define FMCA_INTERPOLATORS_TENSORPRODUCTINTERPOLATOR_H_

namespace FMCA {
/**
 *  \ingroup H2Matrix
 *  \brief Provide functionality for Barycentric Lagrange interpolation
 *         at Chebyshev nodes
 *
 *  The formulas stem from
 *  [1] Berrut, Trefethen: Barycentric Lagrange Interpolation,
 *      SIAM Review 46(3):501-517, 2004*
 *  [2] Higham: The numerical stability of barycentric Lagrange interpolation,
 *      IMA Journal of Numerical Analysis 24:547-556, 2004
 */

/**
 *  \brief These are the classical Chebyshev nodes rescaled to [0,1]
 **/
class TensorProductInterpolator {
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
    const Scalar weight = 2. * FMCA_PI / (2. * Scalar(deg) + 2.);
    // univariate Chebyshev nodes
    xi_ = (weight * (Vector::LinSpaced(deg + 1, 0, deg).array() + 0.5)).cos();
    xi_.array() = 0.5 * (xi_.array() + 1);
    // univariate barycentric weights
    w_ = (weight * (Vector::LinSpaced(deg + 1, 0, deg).array() + 0.5)).sin();
    for (auto i = 1; i < w_.size(); i += 2) w_(i) *= -1.;
    idcs_.init(dim, deg);
    TP_xi_.resize(dim, idcs_.index_set().size());
    // determine tensor product interpolation points
    Index k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < it.size(); ++i) {
        TP_xi_(i, k) = xi_(it[i]);
      }
      ++k;
    }
    V_ = Matrix::Identity(TP_xi_.cols(), TP_xi_.cols());
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  Vector evalPolynomials(const MatrixBase<Derived> &pt) const {
    Vector retval(idcs_.index_set().size());
    Vector weight(dim_);
    Vector my_pt = pt.col(0);
    retval.setOnes();
    Index inf_counter = 0;
    for (auto i = 0; i < dim_; ++i)
      weight(i) = (w_.array() / (my_pt(i) - xi_.array())).sum();
    Index k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < dim_; ++i)
        if (std::abs(my_pt(i) - xi_(it[i])) > FMCA_ZERO_TOLERANCE)
          retval(k) *= w_(it[i]) / (my_pt(i) - xi_(it[i])) / weight(i);
      ++k;
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Matrix &Xi() const { return TP_xi_; }
  const Matrix &invV() const { return V_; }
  const Matrix &V() const { return V_; }

 private:
  MultiIndexSet<TensorProduct> idcs_;
  Matrix TP_xi_;
  Matrix V_;
  Vector xi_;
  Vector w_;
  Index dim_;
  Index deg_;
};
}  // namespace FMCA
#endif

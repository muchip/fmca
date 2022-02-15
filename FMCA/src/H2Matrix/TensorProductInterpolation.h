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
#ifndef FMCA_H2MATRIX_TENSORPRODUCTINTERPOLATION_H_
#define FMCA_H2MATRIX_TENSORPRODUCTINTERPOLATION_H_

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
template <typename ValueType>
class TensorProductInterpolator {
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
    const ValueType weight = 2. * FMCA_PI / (2. * ValueType(deg) + 2.);
    // univariate Chebyshev nodes
    xi_ = (weight * (eigenVector::LinSpaced(deg + 1, 0, deg).array() + 0.5))
              .cos();
    xi_.array() = 0.5 * (xi_.array() + 1);
    // univariate barycentric weights
    w_ = (weight * (eigenVector::LinSpaced(deg + 1, 0, deg).array() + 0.5))
             .sin();
    for (auto i = 1; i < w_.size(); i += 2) w_(i) *= -1.;
    idcs_.init(dim, deg);
    TP_xi_.resize(dim, idcs_.get_MultiIndexSet().size());
    // determine tensor product interpolation points
    IndexType k = 0;
    for (const auto &it : idcs_.get_MultiIndexSet()) {
      for (auto i = 0; i < it.size(); ++i) {
        TP_xi_(i, k) = xi_(it[i]);
      }
      ++k;
    }
    V_ = eigenMatrix::Identity(TP_xi_.cols(), TP_xi_.cols());
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  eigenVector evalLagrangePolynomials(const Eigen::MatrixBase<Derived> &pt) {
    eigenVector retval(idcs_.get_MultiIndexSet().size());
    eigenVector weight(dim_);
    eigenVector my_pt = pt.col(0);
    retval.setOnes();
    IndexType inf_counter = 0;
    for (auto i = 0; i < dim_; ++i)
      weight(i) = (w_.array() / (my_pt(i) - xi_.array())).sum();
    IndexType k = 0;
    for (const auto &it : idcs_.get_MultiIndexSet()) {
      for (auto i = 0; i < dim_; ++i)
        if (abs(my_pt(i) - xi_(it[i])) > FMCA_ZERO_TOLERANCE)
          retval(k) *= w_(it[i]) / (my_pt(i) - xi_(it[i])) / weight(i);
      ++k;
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &Xi() const { return TP_xi_; }
  const eigenMatrix &invV() const { return V_; }
  const eigenMatrix &V() const { return V_; }

 private:
  MultiIndexSet<TensorProduct> idcs_;
  eigenMatrix TP_xi_;
  eigenMatrix V_;
  eigenVector xi_;
  eigenVector w_;
  IndexType dim_;
  IndexType deg_;
};
}  // namespace FMCA
#endif

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
#ifndef FMCA_INTERPOLATOR_EVALPOLYNOMIALS_H_
#define FMCA_INTERPOLATOR_EVALPOLYNOMIALS_H_

namespace FMCA {
namespace internal {

//////////////////////////////////////////////////////////////////////////////
struct LegendrePolynomials {
  template <typename Derived>
  static Matrix eval(Index deg, const MatrixBase<Derived> &pt) {
    Matrix retval(pt.rows(), deg + 1);
    Vector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setZero();
    P1.setOnes();
    retval.col(0) = P1;
    for (Index i = 1; i <= deg; ++i) {
      retval.col(i) =
          Scalar(2 * i - 1) / Scalar(i) * (2 * pt.array() - 1) * P1.array() -
          Scalar(i - 1) / Scalar(i) * P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }
};

//////////////////////////////////////////////////////////////////////////////
struct Monomials {
  template <typename Derived>
  static Matrix eval(Index deg, const MatrixBase<Derived> &pt) {
    Matrix retval(pt.rows(), deg + 1);
    for (Index i = 0; i <= deg; ++i) retval.col(i) = pt.array().pow(i);
    return retval;
  }
};

//////////////////////////////////////////////////////////////////////////////
template <typename Polynomials = LegendrePolynomials, typename MultiIndexSet,
          typename Derived>
Matrix evalPolynomials(const MultiIndexSet &idcs,
                       const MatrixBase<Derived> &pt) {
  Vector retval(idcs.index_set().size());
  Matrix p_values = Polynomials::eval(idcs.max_degree(), pt);
  retval.setOnes();
  Index k = 0;
  for (const auto &it : idcs.index_set()) {
    for (Index i = 0; i < idcs.dim(); ++i) retval(k) *= p_values(i, it[i]);
    ++k;
  }
  return retval;
}

}  // namespace internal
}  // namespace FMCA

#endif

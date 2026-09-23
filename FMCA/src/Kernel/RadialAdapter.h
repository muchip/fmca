// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_KERNEL_RADIALADAPTER_H_
#define FMCA_KERNEL_RADIALADAPTER_H_

namespace FMCA {

/**
 *  \brief metric policies providing the squared distance
 *         s(x, y) entering a radial kernel psi(s).
 **/
struct SquaredEuclidean {
  template <typename Derived, typename otherDerived>
  static Scalar s(const MatrixBase<Derived> &x,
                  const MatrixBase<otherDerived> &y) {
    return (x - y).squaredNorm();
  }
};

/**
 *  \brief anisotropic squared distance s = (x-y)^T A (x-y) for spd A
 **/
struct AnisotropicSquaredEuclidean {
  AnisotropicSquaredEuclidean() {}
  explicit AnisotropicSquaredEuclidean(const Matrix &A) : A_(A) {}
  template <typename Derived, typename otherDerived>
  Scalar s(const MatrixBase<Derived> &x,
           const MatrixBase<otherDerived> &y) const {
    const Vector d = x - y;
    return d.dot(A_ * d);
  }
  Matrix A_;
};

/**
 *  \brief adapter lifting a radial function psi(s, l, c) to a bivariate
 *         kernel evaluable at point pairs, with respect to a metric
 *         policy providing s(x, y).
 **/

template <typename RF, typename Metric = SquaredEuclidean>
struct RadialAdapter {
  typedef RF RadialFunction;
  typedef Metric MetricType;

  static constexpr int cpd_order = RF::cpd_order;
  static constexpr bool has_dpsi = RF::has_dpsi;
  static constexpr bool has_d2psi = RF::has_d2psi;

  RadialAdapter() {}
  explicit RadialAdapter(const Metric &metric) : metric_(metric) {}

  template <typename Derived, typename otherDerived>
  Scalar evaluate(const MatrixBase<Derived> &x,
                  const MatrixBase<otherDerived> &y, Scalar l, Scalar c) const {
    return RF::psi(metric_.s(x, y), l, c);
  }
  Metric metric_;
};

// stateless-metric specialization: fully static evaluate, empty object
template <typename RF>
struct RadialAdapter<RF, SquaredEuclidean> {
  typedef RF RadialFunction;
  typedef SquaredEuclidean MetricType;

  static constexpr int cpd_order = RF::cpd_order;
  static constexpr bool has_dpsi = RF::has_dpsi;
  static constexpr bool has_d2psi = RF::has_d2psi;

  template <typename Derived, typename otherDerived>
  static Scalar evaluate(const MatrixBase<Derived> &x,
                         const MatrixBase<otherDerived> &y, Scalar l,
                         Scalar c) {
    return RF::psi(SquaredEuclidean::s(x, y), l, c);
  }
};

/**
 *  \brief gradient specialization: d/dx_D psi(s) = 2 psi'(s) (x_D - y_D)
 *         is not radial in s, so the adapter computes the chain-rule
 *         factor x_D - y_D itself. Euclidean metric only.
 **/
template <typename RF, int D>
struct RadialAdapter<RadialFunctions::GradientOf<RF, D>, SquaredEuclidean> {
  typedef RF RadialFunction;
  typedef SquaredEuclidean MetricType;

  template <typename Derived, typename otherDerived>
  static Scalar evaluate(const MatrixBase<Derived> &x,
                         const MatrixBase<otherDerived> &y, Scalar l,
                         Scalar c) {
    return 2. * RF::dpsi(SquaredEuclidean::s(x, y), l, c) * (x(D) - y(D));
  }
};

}  // namespace FMCA
#endif

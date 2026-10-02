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
#ifndef FMCA_UTIL_METRIC_H_
#define FMCA_UTIL_METRIC_H_

#include "Macros.h"
namespace FMCA {
namespace Metric {

/**
 *  \brief Euclidean metric. Exposes d(x,y)=\|x-y\|_2 and s(x,y)=d(x,y)^2.
 **/
struct Euclidean {
  template <typename Derived, typename otherDerived>
  static Scalar s(const MatrixBase<Derived> &x,
                  const MatrixBase<otherDerived> &y) {
    return (x - y).squaredNorm();
  }
  template <typename Derived, typename otherDerived>
  static Scalar d(const MatrixBase<Derived> &x,
                  const MatrixBase<otherDerived> &y) {
    return (x - y).norm();
  }
};

/**
 *  \brief Anisotropic euclidean metric. Exposes d(x,y)=\|\sqrt(A)(x-y)\|_2 and
 *         s(x,y)=d(x,y)^2. (not stateless)
 **/
struct AnisotropicEuclidean {
  AnisotropicEuclidean() {}
  explicit AnisotropicEuclidean(const Matrix &A) : A_(A) {}
  template <typename Derived, typename otherDerived>
  Scalar s(const MatrixBase<Derived> &x,
           const MatrixBase<otherDerived> &y) const {
    const Vector diff = x - y;
    return diff.dot(A_ * diff);
  }
  template <typename Derived, typename otherDerived>
  Scalar d(const MatrixBase<Derived> &x,
           const MatrixBase<otherDerived> &y) const {
    return std::sqrt(s(x, y));
  }
  Matrix A_;
};

/**
 *  \brief geodesic distance on the unit sphere d = acos(x^Ty) and
 *         s(x,y)=d(x,y)^2.
 *  **/
struct GeodesicSphere {
  template <typename Derived, typename otherDerived>
  static Scalar d(const MatrixBase<Derived> &x,
                  const MatrixBase<otherDerived> &y) {
    const Scalar clamped_dot = std::min(1., std::max(-1., x.dot(y)));
    return std::acos(clamped_dot);
  }
  template <typename Derived, typename otherDerived>
  static Scalar s(const MatrixBase<Derived> &x,
                  const MatrixBase<otherDerived> &y) {
    const Scalar dist = d(x, y);
    return dist * dist;
  }
};

}  // namespace Metric
}  // namespace FMCA

#endif

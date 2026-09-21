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
#ifndef FMCA_COVARIANCEKERNEL_RADIALFUNCTIONS_H_
#define FMCA_COVARIANCEKERNEL_RADIALFUNCTIONS_H_

#include "../util/Macros.h"

namespace FMCA {
/**
 *  \brief Radial functions in the squared-distance convention
 *         \psi(s) := \phi(\sqrt{s}), s = ||x - y||^2.
 *
 *         Since x -> ||x - y||^2 is smooth, the kernel
 *         k(x, y) = \psi(||x - y||^2) is m-times differentiable across
 *         the diagonal if and only if \psi is C^m at s = 0.
 *
 *         Chain rule for the assembler (Delta = x - y):
 *           d/dx_d k          =  2 dpsi(s) Delta_d
 *           d^2/dx_d dy_e k   = -4 d2psi(s) Delta_d Delta_e
 *                               -2 dpsi(s) delta_{de}
 **/

namespace RadialFunctions {
struct Matern12 {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = false;
  static constexpr bool has_d2psi = false;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(s) / l;
    return std::exp(-arg);
  }
};

struct Matern32 {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = false;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(3. * s) / l;
    return (1. + arg) * std::exp(-arg);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(3. * s) / l;
    return -1.5 / (l * l) * std::exp(-arg);
  }
};

struct Matern52 {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = true;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(5. * s) / l;
    return (1. + (1. + (1. / 3) * arg) * arg) * std::exp(-arg);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(5. * s) / l;
    return -5. / (6. * l * l) * (1. + arg) * std::exp(-arg);
  }
  static Scalar d2psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(5. * s) / l;
    return 25. / (12. * l * l * l * l) * std::exp(-arg);
  }
};

struct Matern72 {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = true;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(7. * s) / l;
    return (1. + (1. + (0.4 + 1. / 15 * arg) * arg) * arg) * std::exp(-arg);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(7. * s) / l;
    return -0.7 / (l * l) * (1. + (1. + (1. / 3) * arg) * arg) * std::exp(-arg);
  }
  static Scalar d2psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = std::sqrt(7. * s) / l;
    return 49. / (60. * l * l * l * l) * (1. + arg) * std::exp(-arg);
  }
};

struct Matern92 {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = true;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = 3. * std::sqrt(s) / l;
    return (1. +
            (1. + (3. / 7 + (2. / 21 + 1. / 105 * arg) * arg) * arg) * arg) *
           std::exp(-arg);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = 3. * std::sqrt(s) / l;
    return -9. / (14. * l * l) *
           (1. + (1. + (0.4 + 1. / 15 * arg) * arg) * arg) * std::exp(-arg);
  }
  static Scalar d2psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = 3. * std::sqrt(s) / l;
    return 81. / (140. * l * l * l * l) * (1. + (1. + (1. / 3) * arg) * arg) *
           std::exp(-arg);
  }
};

struct MaternInf {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = true;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = 0.5 * s / (l * l);
    return std::exp(-arg);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = 0.5 * s / (l * l);
    return -0.5 / (l * l) * std::exp(-arg);
  }
  static Scalar d2psi(Scalar s, Scalar l, Scalar) {
    const Scalar arg = 0.5 * s / (l * l);
    return 0.25 / (l * l * l * l) * std::exp(-arg);
  }
};

struct Multiquadric {
  static constexpr int cpd_order = 1;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = true;
  static Scalar psi(Scalar s, Scalar l, Scalar c) {
    return -std::sqrt(s / (l * l) + c * c);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar c) {
    return -0.5 / (l * l * std::sqrt(s / (l * l) + c * c));
  }
  static Scalar d2psi(Scalar s, Scalar l, Scalar c) {
    const Scalar t = s / (l * l) + c * c;
    return 0.25 / (l * l * l * l * t * std::sqrt(t));
  }
};

struct InvMultiquadric {
  static constexpr int cpd_order = 0;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = true;
  static Scalar psi(Scalar s, Scalar l, Scalar c) {
    return 1. / std::sqrt(s / (l * l) + c * c);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar c) {
    const Scalar t = s / (l * l) + c * c;
    return -0.5 / (l * l * t * std::sqrt(t));
  }
  static Scalar d2psi(Scalar s, Scalar l, Scalar c) {
    const Scalar t = s / (l * l) + c * c;
    return 0.75 / (l * l * l * l * t * t * std::sqrt(t));
  }
};

struct TPS1D {
  static constexpr int cpd_order = 2;
  static constexpr bool has_dpsi = true;
  static constexpr bool has_d2psi = false;
  static Scalar psi(Scalar s, Scalar l, Scalar) {
    const Scalar t = s / (l * l);
    return t * std::sqrt(t);
  }
  static Scalar dpsi(Scalar s, Scalar l, Scalar) {
    return 1.5 * std::sqrt(s) / (l * l * l);
  }
};

struct TPS2D {
  static constexpr int cpd_order = 2;
  static constexpr bool has_dpsi = false;
  static constexpr bool has_d2psi = false;
  static Scalar psi(Scalar s, Scalar, Scalar) {
    return s > 0. ? 0.5 * s * std::log(s) : 0.;
  }
};

struct TPS3D {
  static constexpr int cpd_order = 1;
  static constexpr bool has_dpsi = false;
  static constexpr bool has_d2psi = false;
  static Scalar psi(Scalar s, Scalar l, Scalar) { return -std::sqrt(s) / l; }
};

}  // namespace RadialFunctions
}  // namespace FMCA
#endif

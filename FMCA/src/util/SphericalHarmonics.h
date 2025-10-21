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
#ifndef FMCA_UTIL_SPHERICALHARMONICS_H_
#define FMCA_UTIL_SPHERICALHARMONICS_H_

#include "Macros.h"

namespace FMCA {
struct SphericalHarmonics {
  static constexpr Scalar sqrt2() {
    return 1.414213562373095048801688724209698078569671875376948073176;
  }
  static constexpr Scalar sqrt14pi() {
    return 0.282094791773878143474039725780386292922025314664499428422;
  }
  static constexpr Scalar sqrt4pi() {
    return 3.544907701811032054596334966682290365595098912244774256427;
  }

  /**
   *   \brief evaluates all spherical harmonics from degree 0 to l at x,
   *          which is supposed to be on S^2
   **/
  static Vector Ylm(const Eigen::Vector3d &x, Index deg) {
    // spherical harmonics are defined on the unit sphere only
    assert(std::abs(x.norm() - 1) < 1e-10 && "Ylm: x has to be on S^2");
    Vector retval = Vector::Zero((deg + 1) * (deg + 1));
    Eigen::Vector2d sc;
    sc << 0., 1.;
    Scalar P0 = 0.;
    Scalar P1 = 0.;
    Scalar P2 = 0.;
    Scalar Pdiag = 1.;
    Scalar Klm = 0.;
    // evaluate the associated Legendre polynomials and scale them by (co)sine
    for (int m = 0; m <= deg; ++m) {
      // values of the diagonal entries, the order is important to have the
      // correct scaling for P_0^0.
      retval(m * m) = sc(0) * Pdiag;
      retval(m * m + 2 * m) = sc(1) * Pdiag;
      P0 = 0;
      P1 = Pdiag;
      // values of the other entries in the rows |m|
      for (int l = m + 1; l <= deg; ++l) {
        P2 = ((2. * l - 1.) * P1 * x(2) - (l + m - 1.) * P0) / (Scalar)(l - m);
        // again, the order is important for m = 0;
        retval(l * (l + 1) - m) = sc(0) * P2;
        retval(l * (l + 1) + m) = sc(1) * P2;
        P0 = P1;
        P1 = P2;
      }
      // update sine and cosine
      const Scalar snew = x(0) * sc(0) + x(1) * sc(1);
      const Scalar cnew = x(0) * sc(1) - x(1) * sc(0);
      sc << snew, cnew;
      Pdiag = (1 - 2 * (m + 1)) * Pdiag;
    }
    // scale by Klm and sqrt(2)
    for (int l = 0; l <= deg; ++l) {
      Klm = sqrt(2. * l + 1) * sqrt14pi();
      retval(l * (l + 1)) *= Klm;
      for (int m = 1; m <= l; ++m) {
        Klm /= sqrt(Scalar(l + m) * Scalar(l - m + 1));
        retval(l * (l + 1) - m) *= Klm * sqrt2();
        retval(l * (l + 1) + m) *= Klm * sqrt2();
      }
    }
    return retval;
  }

  static Matrix evaluate(const Matrix &P, Index deg) {
    assert(P.rows() == 3 && "Points have to be in R^3");
    Matrix retval(P.cols(), (deg + 1) * (deg + 1));
#pragma omp parallel for
    for (Index i = 0; i < P.cols(); ++i) retval.row(i) = Ylm(P.col(i), deg);
    return retval;
  }
};
}  // namespace FMCA
#endif

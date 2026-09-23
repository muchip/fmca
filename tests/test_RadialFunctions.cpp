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

#include "../FMCA/src/Kernel/RadialFunctions.h"

template <typename RadialFun>
bool checkRF(FMCA::Scalar l = 0.7, FMCA::Scalar c = 1.3) {
  bool ok = true;
  for (FMCA::Scalar s : {1e-2, 0.1, 1., 10., 100.}) {
    if constexpr (RadialFun::has_dpsi) {
      const FMCA::Scalar h = 1e-4;
      const FMCA::Scalar fd =
          (RadialFun::psi(s + h, l, c) - RadialFun::psi(s - h, l, c)) / (2 * h);
      const FMCA::Scalar err = std::abs(fd - RadialFun::dpsi(s, l, c)) /
                               std::abs(RadialFun::dpsi(s, l, c));
      ok &= err < 1e-4;
    }
    if constexpr (RadialFun::has_d2psi) {
      const FMCA::Scalar h = 1e-4;
      const FMCA::Scalar fd =
          (RadialFun::psi(s + h, l, c) + RadialFun::psi(s - h, l, c) -
           2 * RadialFun::psi(s, l, c)) /
          (h * h);
      const FMCA::Scalar err = std::abs(fd - RadialFun::d2psi(s, l, c)) /
                               std::abs(RadialFun::d2psi(s, l, c));
      ok &= err < 1e-4;
    }
  }
  return ok;
}

template <typename... RadialFuns>
bool checkRadialFunction() {
  return (checkRF<RadialFuns>() && ...);
}

int main() {
  using namespace FMCA::RadialFunctions;
  const bool ok = checkRadialFunction<Matern12, Matern32, Matern52, Matern72,
                                      Matern92, MaternInf, Multiquadric,
                                      InvMultiquadric, TPS1D, TPS2D, TPS3D>();
  if (ok) std::cout << "PASSED" << std::endl;
  return ok ? 0 : 1;
}

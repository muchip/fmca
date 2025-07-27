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
#include <iostream>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/NormalDistribution.h"
#include "../FMCA/src/util/SphericalHarmonics.h"
#include "../FMCA/src/util/Tictoc.h"

FMCA::Matrix FibonacciLattice(const FMCA::Index N) {
  FMCA::Matrix retval(3, N);
  const FMCA::Scalar golden_angle = FMCA_PI * (3.0 - std::sqrt(5.0));

  for (FMCA::Index i = 0; i < N; ++i) {
    const FMCA::Scalar z = 1.0 - (2.0 * i + 1.0) / N;
    const FMCA::Scalar radius = std::sqrt(1.0 - z * z);
    const FMCA::Scalar phi = golden_angle * i;
    const FMCA::Scalar x = radius * std::cos(phi);
    const FMCA::Scalar y = radius * std::sin(phi);
    retval.col(i) << x, y, z;
  }
  return retval;
}

int main() {
  FMCA::Tictoc T;
  const FMCA::Index npts = 1000000;
  const FMCA::Index deg = 100;
  const FMCA::Scalar alpha = .25;
  const FMCA::Matrix P = FibonacciLattice(npts);
  const FMCA::Matrix SH = FMCA::SphericalHarmonics::evaluate(P, deg);
  FMCA::NormalDistribution nd(0, 1);
  FMCA::Vector rdm = nd.randN((deg + 1) * (deg + 1), 1);

  FMCA::Vector sample(npts, 1);
  sample.setZero();

  for (int l = 0; l <= deg; ++l) {
    FMCA::Scalar Cl = std::pow(1.0 + l, -(1.0 * alpha + 1.0));
    for (int m = -l; m <= l; ++m)
      sample += rdm(l * (l + 1) + m) * Cl * SH.col(l * (l + 1) + m);
  }
  FMCA::IO::plotPointsColor("sample.vtk", P, sample);
  return 0;
}

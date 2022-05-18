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
#ifndef FMCA_NUMERICALQUADRATURE_
#define FMCA_NUMERICALQUADRATURE_
namespace FMCA {
namespace Quad {

enum QuadratureType { Midpoint, Trapezoidal, Radon };

template <QuadratureType T> struct Quadrature;

template <> struct Quadrature<Midpoint> {
  Quadrature(void) {
    w.resize(1);
    xi.resize(2, 1);
    w << 0.5;
    xi.col(0) << 1. / 3, 1. / 3;
  }
  Matrix xi;
  Vector w;
};

template <> struct Quadrature<Trapezoidal> {
  Quadrature(void) {
    w.resize(3);
    xi.resize(2, 3);
    w << 1. / 6, 1. / 6, 1. / 6;
    xi << 0, 1, 0, 0, 0, 1;
  }
  Matrix xi;
  Vector w;
};

template <> struct Quadrature<Radon> {
  Quadrature(void) {
    w.resize(7);
    xi.resize(2, 7);
    w << 9. / 80, (155 + sqrt(15)) / 2400, (155 + sqrt(15)) / 2400,
        (155 + sqrt(15)) / 2400, (155 - sqrt(15)) / 2400,
        (155 - sqrt(15)) / 2400, (155 - sqrt(15)) / 2400;
    xi.row(0) << 1. / 3, (6 + sqrt(15)) / 21, (9 - 2 * sqrt(15)) / 21,
        (6 + sqrt(15)) / 21, (6 - sqrt(15)) / 21, (9 + 2 * sqrt(15)) / 21,
        (6 - sqrt(15)) / 21;
    xi.row(1) << 1. / 3, (6 + sqrt(15)) / 21, (6 + sqrt(15)) / 21,
        (9 - 2 * sqrt(15)) / 21, (6 - sqrt(15)) / 21, (6 - sqrt(15)) / 21,
        (9 + 2 * sqrt(15)) / 21;
  }
  Matrix xi;
  Vector w;
};
} // namespace Quad
} // namespace FMCA

#endif

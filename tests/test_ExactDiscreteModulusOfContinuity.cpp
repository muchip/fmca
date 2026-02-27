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
#include "../FMCA/src/ModulusOfContinuity/ExactDiscreteModulusOfContinuity.h"
#include "../FMCA/src/util/Tictoc.h"
#include <Eigen/Dense>
#include <iostream>

int main() {

  FMCA::Tictoc T;

  FMCA::Matrix P(1, 6);
  FMCA::Matrix f1(1, 6);
  FMCA::Matrix f2(1, 6);
  FMCA::Scalar t = 1;
  P << 1, 2, 3, 4, 5, 6;
  f1 << 1, 3, 2, 5, 4, 6;
  f2 << 3, 2, 3, 3, 4, 3;
  FMCA::ExactDiscreteModulusOfContinuity dmoc;
  T.tic();
  dmoc.init(P, f1, "EUCLIDEAN", "EUCLIDEAN");
  std::cout << "omega_t: " << dmoc.computeMocPlot(P, f1, t);
  T.toc("eval: ");

  T.tic();
  dmoc.init(P, f2, "EUCLIDEAN", "EUCLIDEAN");
  std::cout << "omega_t: " << dmoc.computeMocPlot(P, f2, t);
  T.toc("eval: ");
}

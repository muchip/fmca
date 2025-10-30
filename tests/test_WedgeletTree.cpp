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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Wedgelets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 3
#define NPTS 10000

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  const FMCA::Matrix F = Eigen::MatrixXd::Random(NPTS, 1);
  FMCA::WedgeletTree<double> wt(P, F);
  std::cout << wt.bb() << std::endl;
  FMCA::Index i = 0;
  for (const auto &it : wt) std::cout << i++ << " " << it.level() << std::endl;
  return 0;
}

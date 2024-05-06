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

#include "../FMCA/Clustering"
#include "../FMCA/src/Clustering/RandomProjectionTree.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2
#define NPTS 1000

int main() {
  FMCA::Tictoc T;
  FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, 1000);
  FMCA::RandomProjectionTree(P, 10);
  return 0;
}

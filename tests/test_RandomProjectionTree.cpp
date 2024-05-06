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
#define NPTS 1000000

int main() {
  std::srand(std::time(0));
  FMCA::Tictoc T;
  FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  FMCA::Vector colr(NPTS);
  T.tic();
  FMCA::RandomProjectionTree ct(P, 1000);
  T.toc("tree computation: ");
  for (const auto &it : ct) {
    if (!it.nSons()) {
      const FMCA::Index rdm = std::rand() % 256;
      for (FMCA::Index i = 0; i < it.block_size(); ++i)
        colr(it.indices()[i]) = rdm;
    }
  }
  FMCA::Matrix P3D(3, NPTS);
  P3D.setZero();
  P3D.topRows(2) = P;
  FMCA::IO::plotPointsColor("clusters.vtk", P3D, colr);

  return 0;
}

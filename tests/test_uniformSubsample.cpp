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
#include <random>

#include "../FMCA/Clustering"
#include "../FMCA/src/Clustering/uniformSubsample.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  std::mt19937 mt;
  mt.seed(0);
  FMCA::Matrix Psphere(3, 10000);

  {
    std::normal_distribution<FMCA::Scalar> dist(0.0, 1.0);
    for (FMCA::Index i = 0; i < Psphere.cols(); ++i) {
      Psphere.col(i) << dist(mt), dist(mt), dist(mt);
      Psphere.col(i) /= Psphere.col(i).norm();
    }
  }
  FMCA::IO::plotPoints("points00.vtk", Psphere);

  T.tic();
  std::vector<FMCA::Index> idcs;
  std::vector<FMCA::Index> lvls(10);
  for (FMCA::Index i = 0; i < 10; ++i) {
    idcs = FMCA::uniformSubsample(idcs, Psphere, i);
    lvls[i] = idcs.size();
    if (i > 1) std::cout << lvls[i] / lvls[i - 1] << std::endl;
    FMCA::Matrix Psub(3, idcs.size());
    for (FMCA::Index i = 0; i < idcs.size(); ++i)
      Psub.col(i) = Psphere.col(idcs[i]);
    FMCA::ClusterTree ct(Psub, 10);
    FMCA::Vector mdist = FMCA::minDistanceVector(ct, Psub);
    if (i > 1)
      std::cout << mdist.minCoeff() << " " << mdist.maxCoeff() << " "
                << mdist.maxCoeff() / mdist.minCoeff() << std::endl;
    FMCA::IO::plotPoints("points" + std::to_string(i) + ".vtk", Psub);
  }
  T.toc("generated levels");
}

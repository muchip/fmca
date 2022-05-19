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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/Tictoc.h"
#include "TestParameters.h"

int main() {
  FMCA::Tictoc T;
  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = 1. / 0.;
  const FMCA::Index n = 50000;
  const FMCA::Matrix P = FMCA::Matrix::Random(3, n);
  T.tic();
  for (auto j = 0; j < P.cols(); ++j) {
    FMCA::Scalar dist = FMCA::Scalar(1. / 0.);
    for (auto i = 0; i < P.cols(); ++i) {
      if (i != j) {
        FMCA::Scalar rad = 0.5 * (P.col(i) - P.col(j)).norm();
        dist = dist > (2 * rad) ? (2 * rad) : dist;
        separation_radius = separation_radius < rad ? separation_radius : rad;
      }
    }
    fill_distance = fill_distance < dist ? dist : fill_distance;
  }
  T.toc("dist mat:         ");
  std::cout << "fill_distance:     " << fill_distance << std::endl;
  std::cout << "separation_radius: " << separation_radius << std::endl;
  std::cout
      << "Cluster splitter:  "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  for (auto i = 0; i < 10; ++i) {
    FMCA::Index leaf_size = rand() % 200 + 5;
    T.tic();
    FMCA::ClusterTree CT(P, leaf_size);
    assert(abs(fill_distance - FMCA::fillDistance(CT, P)) <
           FMCA_ZERO_TOLERANCE);
    assert(abs(separation_radius - FMCA::separationRadius(CT, P)) <
           FMCA_ZERO_TOLERANCE);
    T.toc("cluster tree:     ");
    for (auto &&it : CT) {
      if (!it.nSons())
        for (auto j = 0; j < it.indices().size(); ++j)
          assert(FMCA::inBoundingBox(it, P.col(it.indices()[j])) &&
                 "point outside leaf bounding box");
    }
  }

  return 0;
}

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

#include <FMCA/Clustering>
#include <FMCA/src/util/Tictoc.h>

int main() {
  const FMCA::Index N = 10000;
  const FMCA::Matrix P = FMCA::Matrix::Random(3, N);
  FMCA::Tictoc T;
  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = 1. / 0.;

  T.tic();
  const FMCA::ClusterTree CT(P, 10);
  T.toc("cluster tree: ");

  T.tic();
  for (auto j = 0; j < P.cols(); ++j) {
    FMCA::Scalar dist = 1. / 0.;
    for (auto i = 0; i < P.cols(); ++i) {
      if (i != j) {
        FMCA::Scalar rad = 0.5 * (P.col(i) - P.col(j)).norm();
        dist = dist > (2 * rad) ? (2 * rad) : dist;
        separation_radius = separation_radius < rad ? separation_radius : rad;
      }
    }
    fill_distance = fill_distance < dist ? dist : fill_distance;
  }
  std::cout << "fill_distance:     " << fill_distance << std::endl;
  std::cout << "separation_radius: " << separation_radius << std::endl;
  T.toc("dist mat: ");

  T.tic();
  std::cout << "error fd: " << abs(fill_distance - FMCA::fillDistance(CT, P))
            << std::endl;
  std::cout << "error sr: "
            << abs(separation_radius - FMCA::separationRadius(CT, P))
            << std::endl;
  T.toc("computation time: ");

  return 0;
}

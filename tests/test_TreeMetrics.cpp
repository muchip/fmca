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
  double fill_distance = 0;
  double separation_radius = 1. / 0.;
  const unsigned int n = 160000;
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(3, n);
  T.tic();
  for (auto j = 0; j < P.cols(); ++j) {
    double dist = double(1.) / double(0.);
    for (auto i = 0; i < P.cols(); ++i) {
      if (i != j) {
        double rad = 0.5 * (P.col(i) - P.col(j)).norm();
        dist = dist > (2 * rad) ? (2 * rad) : dist;
        separation_radius = separation_radius < rad ? separation_radius : rad;
      }
    }
    fill_distance = fill_distance < dist ? dist : fill_distance;
  }
  T.toc("dist mat: ");

  std::cout << "fill_distance:     " << fill_distance << std::endl;
  std::cout << "separation_radius: " << separation_radius << std::endl;
  T.tic();
  const FMCA::ClusterTree CT(P, 10);
  T.toc("cluster tree: ");
  T.tic();
  std::cout << "error fd: " << abs(fill_distance - FMCA::fillDistance(CT, P))
            << std::endl;
  std::cout << "error sr: "
            << abs(separation_radius - FMCA::separationRadius(CT, P))
            << std::endl;
  T.toc("computation time: ");

  return 0;
}

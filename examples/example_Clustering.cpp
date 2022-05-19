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
#include <FMCA/Clustering>
#include <FMCA/src/util/Tictoc.h>
#include <FMCA/src/util/print2file.h>
#include <iomanip>
#include <iostream>

int main() {
  const FMCA::Index d = 10;
  const FMCA::Index N = 10000;
  const FMCA::Matrix P = FMCA::Matrix::Random(d, N);
  FMCA::Tictoc T;
  std::cout << "dimension:                  " << d << std::endl;
  std::cout << "number of points:           " << N << std::endl;
  T.tic();
  FMCA::ClusterTree CT(P, 666);
  FMCA::IO::print2m("cluster1D.m", "P", P, "w");
  FMCA::Matrix bb(std::distance(CT.begin(), CT.end()), 3);
  FMCA::Index i = 0;
  for (auto &&it : CT) {
    bb.row(i) << (it.nSons() > 0), it.bb()(0, 0), it.bb()(0, 1);
    ++i;
  }
  FMCA::IO::print2m("cluster1D.m", "BB", bb, "a");
  T.toc("cluster tree assembly time:");
  double fill_distance = 0;
  double separation_radius = 1. / 0.;
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
  std::cout << fill_distance << " " << separation_radius << std::endl;
  FMCA::clusterTreeStatistics(CT, P);

  return 0;
}

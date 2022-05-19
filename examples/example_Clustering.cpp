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
#include <iomanip>
#include <iostream>

int main() {
  const FMCA::Index d = 3;
  const FMCA::Index N = 1000;
  const FMCA::Matrix P = FMCA::Matrix::Random(d, N);
  FMCA::Tictoc T;
  std::cout << "dimension:                  " << d << std::endl;
  std::cout << "number of points:           " << N << std::endl;
  T.tic();
  FMCA::ClusterTree CT(P, 10);
  T.toc("cluster tree assembly time:");
  FMCA::clusterTreeStatistics(CT, P);

  return 0;
}

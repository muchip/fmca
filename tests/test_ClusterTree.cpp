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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 3
#define NPTS 10000000

int main() {
  FMCA::Tictoc T;

  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = FMCA_INF;
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
#if 0
#pragma omp parallel for
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
  std::cout << "fill_distance:                " << fill_distance << std::endl;
  std::cout << "separation_radius:            " << separation_radius
            << std::endl;
#endif
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.tic();
  for (auto i = 0; i < 10; ++i) {
    FMCA::iVector index_hits(P.cols());
    index_hits.setZero();
    FMCA::Index leaf_size = rand() % 200 + 5;
    FMCA::ClusterTree CT(P, leaf_size);
    for (i = 0; i < P.cols(); ++i) index_hits(CT.indices()[i]) = 1;
    assert(index_hits.sum() == P.cols() && "CT lost indices");
  }
  T.toc("construction of 10 cluster trees: ");
  return 0;
}

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

#define DIM 3
#define NPTS 10000

int main() {
  FMCA::Tictoc T;
  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = FMCA_INF;
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  for (auto j = 0; j < P.cols(); ++j) {
    FMCA::Scalar dist = FMCA_INF;
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
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  for (auto i = 0; i < 10; ++i) {
    FMCA::Index leaf_size = rand() % 200 + 5;
    FMCA::ClusterTree CT(P, leaf_size);
    FMCA::Vector min_dist = minDistanceVector(CT, P);
    FMCA::Scalar fill_distance_test = min_dist.maxCoeff();
    FMCA::Scalar separation_radius_test = 0.5 * min_dist.minCoeff();
    assert(abs(fill_distance - fill_distance_test) < FMCA_ZERO_TOLERANCE);
    assert(abs(separation_radius - separation_radius_test) <
           FMCA_ZERO_TOLERANCE);
    for (auto &&it : CT) {
      if (!it.nSons())
        for (auto j = 0; j < it.indices().size(); ++j)
          assert(FMCA::inBoundingBox(it, P.col(it.indices()[j])) &&
                 "point outside leaf bounding box");
    }
  }

  return 0;
}

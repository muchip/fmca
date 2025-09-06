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
#include "../FMCA/src/ModulusOfContinuity/greedySetCovering.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  const FMCA::Index nPts = 1000000;
  const FMCA::Scalar r = .05;
  FMCA::Tictoc T;
  std::mt19937 mt;
  mt.seed(0);
  FMCA::Matrix Psphere(3, nPts);
  FMCA::Matrix Psquare(2, nPts);

  {
    std::normal_distribution<FMCA::Scalar> dist(0.0, 1.0);
    for (FMCA::Index i = 0; i < Psphere.cols(); ++i) {
      Psphere.col(i) << dist(mt), dist(mt), dist(mt);
      Psphere.col(i) /= Psphere.col(i).norm();
    }
    Psquare.setRandom();
  }

  T.tic();
  FMCA::KDTree ct(Psquare, 10);
  std::vector<FMCA::Index> idcs = FMCA::greedySetCovering(ct, Psquare, r);
  T.toc("covering square: ");
  T.tic();
  FMCA::SphereClusterTree sct(Psphere, 10);
  std::vector<FMCA::Index> sidcs = FMCA::greedySetCovering(sct, Psphere, r);
  T.toc("covering sphere: ");

  // test square cover
  {
    std::vector<bool> is_covered(nPts, false);
    std::vector<FMCA::Index> cover_index(nPts, -1);
    FMCA::Vector color(nPts);
#pragma omp parallel for
    for (FMCA::Index i = 0; i < nPts; ++i) {
      FMCA::Scalar min_dist = FMCA_INF;
      for (FMCA::Index j = 0; j < idcs.size(); ++j) {
        const FMCA::Scalar dist =
            (Psquare.col(i) - Psquare.col(idcs[j])).norm();
        if (dist < 0.5 * r) {
          is_covered[i] = true;
          cover_index[i] = min_dist > dist ? idcs[j] : cover_index[i];
          min_dist = min_dist > dist ? dist : min_dist;
        }
      }
      color(i) = (Psquare.col(i) - Psquare.col(cover_index[i])).norm();
      assert(is_covered[i] && "point not covered");
      assert(color(i) < 0.5 * r && "covering distance too large");
    }
    FMCA::Matrix P3(3, nPts);
    P3.setZero();
    P3.topRows(2) = Psquare;
    FMCA::IO::plotPointsColor("squareCover.vtk", P3, color);
  }
  // test sphere cover
  {
    std::vector<bool> is_covered(nPts, false);
    std::vector<FMCA::Index> cover_index(nPts, -1);
    FMCA::Vector color(nPts);
#pragma omp parallel for
    for (FMCA::Index i = 0; i < nPts; ++i) {
      FMCA::Scalar min_dist = FMCA_INF;
      for (FMCA::Index j = 0; j < sidcs.size(); ++j) {
        const FMCA::Scalar dist =
            sct.geodesicDistance(Psphere.col(i), Psphere.col(sidcs[j]));
        if (dist < 0.5 * r) {
          is_covered[i] = true;
          cover_index[i] = min_dist > dist ? sidcs[j] : cover_index[i];
          min_dist = min_dist > dist ? dist : min_dist;
        }
      }
      color(i) =
          sct.geodesicDistance(Psphere.col(i), Psphere.col(cover_index[i]));
      assert(is_covered[i] && "point not covered");
      assert(color(i) < 0.5 * r && "covering distance too large");
    }
    FMCA::IO::plotPointsColor("sphereCover.vtk", Psphere, color);
  }

  return 0;
}

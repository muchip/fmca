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
  {
    // first test priority queue (AI written test)
    const FMCA::Index N = 5;
    FMCA::PriorityQueue pq(N);

    // push (key, index)
    // keys: 0->10, 1->5, 2->7, 3->3, 4->8
    pq.push(10, 0);
    pq.push(5, 1);
    pq.push(7, 2);
    pq.push(3, 3);
    pq.push(8, 4);

    // top should be (10, 0)
    {
      auto [k, i] = pq.top();
      assert(k == 10 && i == 0);
    }

    // decrease key of index 0 by 1: 10 -> 9
    pq.decreaseKey(0);
    {
      auto [k, i] = pq.top();
      // still (9, 0) should be the max
      assert(k == 9 && i == 0);
    }

    // decrease index 0 several times to move it below others
    for (int t = 0; t < 5; ++t) pq.decreaseKey(0);  // 9 -> 4
    {
      auto [k, i] = pq.top();
      // keys now: 0->4, 1->5, 2->7, 3->3, 4->8
      // max is (8, 4)
      assert(k == 8 && i == 4);
    }

    // pop max (8, 4)
    pq.pop();
    {
      auto [k, i] = pq.top();
      // remaining keys: 0->4, 1->5, 2->7, 3->3
      // max is (7, 2)
      assert(k == 7 && i == 2);
    }

    // decrease index 2 until its key reaches 0
    for (int t = 0; t < 10; ++t) pq.decreaseKey(2);
    {
      auto [k, i] = pq.top();
      // keys: 0->4, 1->5, 2->0, 3->3  => max is (5, 1)
      assert(k == 5 && i == 1);
    }

    // pop all and check that keys are non-increasing
    std::vector<FMCA::Index> keys;
    while (!pq.empty()) {
      auto [k, i] = pq.top();
      keys.push_back(k);
      pq.pop();
    }
    for (std::size_t j = 1; j < keys.size(); ++j) {
      assert(keys[j - 1] >= keys[j]);
    }

    std::cout << "PriorityQueue tests passed.\n";
  }
  const FMCA::Index nPts = 100000;
  const FMCA::Scalar r = .01;
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

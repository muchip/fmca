// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <metis.h>

#include <Eigen/Dense>
#include <iostream>
#include <random>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/MDS.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  const FMCA::Index d = 3;
  const FMCA::Index n = 1000;

  const FMCA::Matrix P = FMCA::Matrix::Random(d, n);
  T.tic();
  const FMCA::ClusterTree CT(P, 10);
  T.toc("cluster tree: ");
  for (FMCA::Index knn : {10, 50, 100, 500, 600, 700, 800, 900, 999}) {
    T.tic();
    std::vector<FMCA::Triplet> A = FMCA::symKNN(CT, P, knn);
    T.toc("kNN:");
    T.tic();
    FMCA::Graph<idx_t, FMCA::Scalar> G;
    G.init(n, A);
    FMCA::Matrix D = G.distanceMatrix();
    T.toc("construct graph");
    FMCA::Matrix Pemb = FMCA::MDS(D, d);
    FMCA::Matrix Demb(n, n);
    for (FMCA::Index j = 0; j < n; ++j)
      for (FMCA::Index i = 0; i < n; ++i)
        Demb(i, j) = (Pemb.col(i) - Pemb.col(j)).norm();
    std::cout << "lost energy: " << (Demb - D).norm() / D.norm() << std::endl;
  }
  return 0;
}

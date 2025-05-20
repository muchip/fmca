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
#include <metis.h>

#include <Eigen/Dense>
#include <iostream>
#include <random>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 3
#define NPTS 1000

int main() {
  FMCA::Tictoc T;
  const FMCA::Index npts = 1000000;
  const FMCA::Index leaf_size = 100;

  std::mt19937 mt;
  mt.seed(0);
  FMCA::Matrix P(3, npts);
  {
    std::normal_distribution<FMCA::Scalar> dist(0.0, 1.0);
    for (FMCA::Index i = 0; i < P.cols(); ++i) {
      P.col(i) << dist(mt), dist(mt), dist(mt);
      P.col(i) /= P.col(i).norm();
    }
  }
  T.tic();
  FMCA::ClusterTree CT(P, 100);
  std::vector<Eigen::Triplet<FMCA::Scalar>> A =
      FMCA::symKNN(CT, P, 10);
  T.toc("kNN:");

  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(npts, A);

  FMCA::MetisClusterTree MCT(G, leaf_size);
  FMCA::Vector color(P.cols());
  FMCA::iVector labels(P.cols());
  color.setZero();
  labels.setZero();
  for (auto &&it : MCT) {
    if (!it.nSons()) {
      FMCA::Scalar clr = rand() % 1000;
      for (FMCA::Index i = 0; i < it.block_size(); ++i) {
        color(it.indices()[i]) = clr;
        labels(it.indices()[i]) = 1;
      }
    }
  }
  assert(labels.sum() == P.cols() && "missing or duplicate labels");
  FMCA::IO::plotPointsColor("points.vtk", P, color);
  return 0;
}

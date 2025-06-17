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
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/GraphSampletTree.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 2
#define NPTS 1000

using SampletInterpolator = FMCA::TotalDegreeInterpolator;
using SampletTree = FMCA::GraphSampletTree;

int main() {
  FMCA::Tictoc T;
  const FMCA::Index npts = 100000;
  const FMCA::Index leaf_size = 10;

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
  FMCA::ClusterTree CT(P, 200);
  T.toc("RPT: ");
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, 100);
  T.toc("kNN:");

  T.tic();
  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(npts, A);
  T.toc("construct graph");
  FMCA::TotalDegreeInterpolator interp;
  interp.init(2, 4);
  SampletTree st(interp, 10, G);
  return 0;
}

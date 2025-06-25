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
extern "C" {
#include <metis.h>
}

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <random>
//
#include "../FMCA/Clustering"
#include "../FMCA/src/util/Graph.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 100000;
  const FMCA::Index kNN = 10;
  const FMCA::Index M = 10;
#if 1
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
#if 1
  {
    for (FMCA::Index i = 0; i < npts; ++i) {
      const FMCA::Scalar u = FMCA::Scalar(rand()) / RAND_MAX;
      const FMCA::Scalar v = FMCA::Scalar(rand()) / RAND_MAX;
      const FMCA::Scalar t = 1.5 * FMCA_PI + 4.5 * FMCA_PI * u;
      const FMCA::Scalar x = t * std::cos(t);
      const FMCA::Scalar y = 21.0 * v;
      const FMCA::Scalar z = t * std::sin(t);
      P.col(i) << x, y, z;
    }
  }
#endif
#else
  const FMCA::Matrix data =
      FMCA::IO::ascii2Matrix("era5_20-06-24-12h.txt").transpose();
  FMCA::Matrix P = data.topRows(3);
  FMCA::Matrix temp = data.row(3);
#endif
  FMCA::ClusterTree CT(P, kNN);
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, kNN);
  T.toc("kNN:");

  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(P.cols(), A);
  G.print("graph0.vtk", P);
  auto lm = G.computeLandmarkNodes(100);
  FMCA::Matrix LM(3, 100);
  for (FMCA::Index i = 0; i < LM.cols(); ++i) LM.col(i) = P.col(lm[i]);
  FMCA::IO::plotPoints("landmarks.vtk", LM);
  T.tic();
  auto part = FMCA::METIS::partitionGraphKWay(G, M);
  FMCA::Vector v(part.size());
  for (FMCA::Index i = 0; i < v.size(); ++i) v(i) = part[i];
  T.toc("metis part: ");
  FMCA::IO::plotPointsColor("points.vtk", P, v);
  return 0;
}

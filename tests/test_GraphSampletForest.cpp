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
extern "C" {
#include <metis.h>
}

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <random>
//
#include "../FMCA/Samplets"
#include "../FMCA/src/util/LIsomap.h"
#include "../FMCA/src/util/Tictoc.h"

using Graph = FMCA::Graph<idx_t, FMCA::Scalar>;

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 100000;
  const FMCA::Index k = 20;

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
  FMCA::Vector signal(P.cols());
  for (FMCA::Index i = 0; i < P.cols(); ++i)
    signal(i) = P(0, i) * P(0, i) + P(1, i) + P(2, i);
  FMCA::ClusterTree CT(P, 100);
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, k);
  T.toc("kNN:");
  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(P.cols(), A);
  T.tic();
  FMCA::GraphSampletForest<Graph> gsf(G, 10, 2, 3);
  T.toc("samplet forest: ");
  for (FMCA::Index i = 0; i < 6; ++i)
    std::cout << gsf.lost_energies()[i] << std::endl;
  FMCA::Vector Tsignal = gsf.sampletTransform(signal);
  FMCA::Index nneg = 0;
  for (FMCA::Index i = 0; i < P.cols(); ++i) {
    if (std::abs(Tsignal(i) * Tsignal(i)) > Tsignal.norm() * 1e-4) ++nneg;
  }
  Tsignal = gsf.inverseSampletTransform(Tsignal);
  std::cout << "isometry? " << (Tsignal - signal).norm() << std::endl;
  std::cout << "non negligible coeffs: " << nneg << std::endl;

  return 0;
}

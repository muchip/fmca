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

using SampletInterpolator = FMCA::TotalDegreeInterpolator;
using SampletTree = FMCA::GraphSampletTree;

int main() {
  FMCA::Tictoc T;
  const FMCA::Index npts = 1000000;
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
  FMCA::ClusterTree CT(P, 10);
  T.toc("RPT: ");
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, 100);
  T.toc("kNN:");

  T.tic();
  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(npts, A);
  T.toc("construct graph");
  FMCA::MonomialInterpolator interp;
  interp.init(2, 2);
  FMCA::Vector signal(P.cols());
  for (FMCA::Index i = 0; i < signal.size(); ++i) signal(i) = P(0, i) * P(1, i);
  SampletTree st(interp, 20, G);
  FMCA::Matrix I(npts, npts);
  I.setIdentity();
  // FMCA::Matrix S = st.sampletTransform(I);
  // std::cout << "err: " << (S * S.transpose() - I).norm() / I.norm() << " "
  //           << (S.transpose() * S - I).norm() / I.norm() << std::endl;
  FMCA::Vector Ss = st.sampletTransform(st.toClusterOrder(signal));
  int ctr = 0;
  for (FMCA::Index i = 0; i < Ss.size(); ++i)
    if (abs(Ss(i) / Ss.norm()) > 1e-4) {
      std::cout << i << " " << Ss(i) << std::endl;
      ++ctr;
    }
  std::cout << "non-negligible: " << ctr << std::endl;
  return 0;
}

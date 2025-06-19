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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

using SampletInterpolator = FMCA::TotalDegreeInterpolator;
using SampletTree = FMCA::GraphSampletTree;

int main() {
  FMCA::Tictoc T;
  const FMCA::Index npts = 100000;
  const FMCA::Index leaf_size = 200;

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
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, 500);
  T.toc("kNN:");

  T.tic();
  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(npts, A);
  T.toc("construct graph");
  FMCA::MonomialInterpolator interp;
  interp.init(2, 4);
  FMCA::Vector signal(P.cols());
  for (FMCA::Index i = 0; i < signal.size(); ++i)
    signal(i) = P(0, i) * P(1, i) + P(2, i) * P(2, i);
  G.printSignal("data_graph.vtk", P, signal);
  SampletTree st(interp, leaf_size, G);
  FMCA::Matrix I(npts, npts);
  I.setIdentity();
  // FMCA::Matrix S = st.sampletTransform(I);
  // std::cout << "err: " << (S * S.transpose() - I).norm() / I.norm() << " "
  //           << (S.transpose() * S - I).norm() / I.norm() << std::endl;
  FMCA::Vector Ss = st.sampletTransform(st.toClusterOrder(signal));
  int ctr = 0;
  for (FMCA::Index i = 0; i < Ss.size(); ++i)
    if (abs(Ss(i) / Ss.norm()) > 1e-2) {
      std::cout << i << " " << Ss(i) << std::endl;
      ++ctr;
    }
  std::cout << "non-negligible: " << ctr << std::endl;
  FMCA::Index f = 0;
  for (const auto &it : st) {
    if (!it.nSons()) {
      FMCA::Vector sig(it.block_size());
      for (FMCA::Index i = 0; i < it.block_size(); ++i)
        sig(i) = signal(it.indices()[i]);
      FMCA::Matrix P3(3, it.node().P.cols());
      P3.setZero();
      P3.topRows(2) = it.node().P;
      FMCA::Vector minP = P3.rowwise().minCoeff();
      FMCA::Vector maxP = P3.rowwise().maxCoeff();
      FMCA::Vector inv_dist = 1. / (maxP - minP).array();
      inv_dist(2) = 0;
      P3 = inv_dist.asDiagonal() *
           (P3 - minP * FMCA::Vector::Ones(P3.cols()).transpose());
      P3.bottomRows(1) = sig.transpose();
      std::vector<Eigen::Triplet<FMCA::Scalar>> trips;
      trips.reserve(it.block_size() * it.block_size());
      for (FMCA::Index i = 0; i < it.block_size(); ++i)
        for (FMCA::Index j = 0; j < i; ++j) {
          const FMCA::Scalar w =
              G.graph().coeff(it.indices()[i], it.indices()[j]);
          if (std::abs(w) > FMCA_ZERO_TOLERANCE) {
            trips.push_back(Eigen::Triplet<FMCA::Scalar>(i, j, w));
            trips.push_back(Eigen::Triplet<FMCA::Scalar>(j, i, w));
          }
        }
      FMCA::Graph<idx_t, FMCA::Scalar> G2;
      G2.init(it.block_size(), trips);
      G2.printSignal("leaf_flat" + std::to_string(f) + ".vtk", P3, sig);
      for (FMCA::Index i = 0; i < it.block_size(); ++i)
        P3.col(i) = P.col(it.indices()[i]);
      G2.printSignal("leaf" + std::to_string(f) + ".vtk", P3, sig);

      ++f;
    }
  }
  return 0;
}

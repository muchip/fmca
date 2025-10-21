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
  const FMCA::Index npts = 1000;
  const FMCA::Index tpts = 8;
  const FMCA::Index leaf_size = 3;

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
  P.setZero();
  P.topRows(2).setRandom();
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
  SampletTree st;
  st.init<FMCA::TotalDegreeInterpolator, FMCA::Graph<idx_t, FMCA::Scalar>>(
      G, 2, leaf_size, 2);
  FMCA::Matrix sP = P;
  for (FMCA::Index i = 0; i < sP.cols(); ++i)
    sP.col(i) = P.col(st.indices()[i]);
  {
    FMCA::Matrix test(npts, tpts);
    std::vector<FMCA::Index> indices(npts);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    test.setZero();
    for (FMCA::Index i = 0; i < tpts; ++i) {
      test(indices[i], i) = 1;
    }
    const FMCA::Matrix samp_basis =
        st.inverseSampletTransform(FMCA::Matrix::Identity(npts, npts));
    for (FMCA::Index i = 0; i < 1000; ++i)
      G.printSignal("global_sampl" + std::to_string(i) + ".vtk", sP,
                    samp_basis.col(i));
    const FMCA::Matrix sample_cols = st.inverseSampletTransform(test);
    const FMCA::Matrix test_I = sample_cols.transpose() * sample_cols;

    std::cout << "orthogonality error: "
              << (test_I - FMCA::Matrix::Identity(tpts, tpts)).norm() /
                     std::sqrt(tpts)
              << std::endl;
  }
#if 0
  FMCA::Vector unit(npts);
  unit.setZero();
  unit(0) = 1;
  G.printSignal("global_sampl" + std::to_string(0) + ".vtk", P,
                st.sampletTransform(unit));
  unit(0) = 0;
  unit(100) = 1;
  G.printSignal("global_sampl" + std::to_string(1) + ".vtk", P,
                st.sampletTransform(unit));
  unit(100) = 0;
  unit(1000) = 1;
  G.printSignal("global_sampl" + std::to_string(2) + ".vtk", P,
                st.sampletTransform(unit));
  // FMCA::Matrix S = st.sampletTransform(I);
  // std::cout << "err: " << (S * S.transpose() - I).norm() / I.norm() << " "
  //           << (S.transpose() * S - I).norm() / I.norm() << std::endl;
  FMCA::Vector Ss = st.sampletTransform(st.toClusterOrder(signal));
  int ctr = 0;
  for (FMCA::Index i = 0; i < Ss.size(); ++i)
    if (abs(Ss(i) / Ss.norm()) > 1e-4) {
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
      if (f < 2) {
        FMCA::Matrix P3(3, it.node().P.cols());
        P3.setZero();
        P3.topRows(2) = it.node().P;
        FMCA::Vector minP = P3.rowwise().minCoeff();
        FMCA::Vector maxP = P3.rowwise().maxCoeff();
        FMCA::Vector inv_dist = 1. / (maxP - minP).array();
        inv_dist(2) = 0;
        P3 = inv_dist.asDiagonal() *
             (P3 - minP * FMCA::Vector::Ones(P3.cols()).transpose());
        for (FMCA::Index i = 0; i < it.Q().cols(); ++i) {
          P3.bottomRows(1) = it.Q().col(i).transpose();
          G2.printSignal("samplets" + std::to_string(i) + ".vtk", P3,
                         it.Q().col(i));
        }
      }
    }
  }
#endif
  return 0;
}

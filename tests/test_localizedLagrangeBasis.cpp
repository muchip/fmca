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
#include <fstream>
#include <iostream>

#include "../FMCA/Clustering"
#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/src/Clustering/epsNN.h"
#include "../FMCA/src/util/Tictoc.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 20000;
  const FMCA::Index dim = 3;
  const FMCA::Index leaf_size = 100;
  const FMCA::Index K = 2;
  const FMCA::Index mpole_deg = 4;
  const FMCA::Scalar eta = 1. / dim;
  const FMCA::CovarianceKernel kernel("EXPONENTIAL", 1.);
  const FMCA::Matrix P = FMCA::Matrix::Random(dim, npts);

  T.tic();
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, kernel);
  H2ClusterTree CT(mom, 0, P);
  T.toc("cluster tree:                ");
  std::vector<FMCA::Index> inv_idcs(P.cols());
  for (FMCA::Index i = 0; i < inv_idcs.size(); ++i)
    inv_idcs[CT.indices()[i]] = i;
  T.tic();
  const FMCA::Vector mdv = FMCA::minDistanceVector(CT, P);
  T.toc("min dist:                    ");
  const FMCA::Scalar fill_distance = mdv.maxCoeff();
  const FMCA::Scalar search_radius =
      K * fill_distance * std::abs(std::log(fill_distance));
  std::cout << "fill distance:                " << fill_distance << std::endl;
  std::cout << "search radius:                " << search_radius << std::endl;
  std::vector<std::vector<FMCA::Index>> epsnn(P.cols());
  T.tic();
#pragma omp parallel for
  for (FMCA::Index i = 0; i < epsnn.size(); ++i)
    epsnn[i] = FMCA::epsNN(CT, P, P.col(i), search_radius);
  T.toc("epsnn:                       ");
  T.tic();
  std::cout << std::string(60, '-') << std::endl;
  std::vector<Eigen::Triplet<FMCA::Scalar>> triplets;

#pragma omp parallel for
  for (FMCA::Index i = 0; i < epsnn.size(); ++i) {
    const FMCA::Index locN = epsnn[i].size();
    FMCA::Matrix Ploc(dim, locN);
    FMCA::Index pos = 0;
    for (FMCA::Index j = 0; j < locN; ++j) {
      Ploc.col(j) = P.col(epsnn[i][j]);
      pos = epsnn[i][j] == i ? j : pos;
    }
    FMCA::Matrix Kloc = kernel.eval(Ploc, Ploc);
    FMCA::Vector rhs(locN);
    rhs.setZero();
    rhs(pos) = 1;
    FMCA::Vector col = Kloc.ldlt().solve(rhs);
    std::vector<Eigen::Triplet<FMCA::Scalar>> local_triplets;
    for (FMCA::Index j = 0; j < locN; ++j)
      local_triplets.push_back(Eigen::Triplet<FMCA::Scalar>(
          inv_idcs[i], inv_idcs[epsnn[i][j]], col(j)));
#pragma omp critical
    triplets.insert(triplets.end(), local_triplets.begin(),
                    local_triplets.end());
  }
  T.toc("local inverse triplets:      ");
  Eigen::SparseMatrix<FMCA::Scalar> invK(npts, npts);
  invK.setFromTriplets(triplets.begin(), triplets.end());
  std::cout << "anz:                          " << triplets.size() / npts
            << std::endl;
  {
    FMCA::Matrix X(npts, 10), Y1(npts, 10), Y2(npts, 10);
    X.setZero();
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      FMCA::Vector col = kernel.eval(P, P.col(CT.indices()[index]));
      Y1.col(i) =
          col(Eigen::Map<const FMCA::iVector>(CT.indices(), CT.block_size()));
      X(index, i) = 1;
    }
    Y2 = invK * Y1;
    FMCA::Scalar err = (Y2 - X).norm() / X.norm();
    std::cout << "inverse error:                " << err << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  T.tic();
  const H2Matrix hmat(CT, mat_eval, eta);
  T.toc("elapsed time H2-matrix:      ");
  hmat.statistics();
  {
    FMCA::Matrix X(npts, 10), Y1(npts, 10), Y2(npts, 10);
    X.setZero();
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      FMCA::Vector col = kernel.eval(P, P.col(CT.indices()[index]));
      Y1.col(i) =
          col(Eigen::Map<const FMCA::iVector>(CT.indices(), CT.block_size()));
      X(index, i) = 1;
    }
    Y2 = hmat * X;
    FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
    std::cout << "compression error:            " << err << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  hmat.statistics();

  FMCA::Matrix X(npts, 10), Y1(npts, 10), Y2(npts, 10);
  X.setRandom();
  Y1 = hmat * X;
  Y2 = invK * Y1;
  std::cout << "inverse error vs H2-Matrix:   " << (Y2 - X).norm() / X.norm()
            << std::endl;
  return 0;
}

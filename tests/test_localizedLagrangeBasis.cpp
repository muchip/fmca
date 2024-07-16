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
#include <fstream>
#include <iostream>
//
#include <Eigen/Dense>
// #include <Eigen/MetisSupport>
// #include <Eigen/Sparse>
// #include <Eigen/SparseCholesky>
// using EigenCholesky =
//     Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
//                          Eigen::MetisOrdering<int>>;

#include "../FMCA/Clustering"
#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/HMatrix"
#include "../FMCA/src/Clustering/epsNN.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/IO.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using HMatrix = FMCA::HMatrix<H2ClusterTree>;

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 100;
  const FMCA::Index dim = 1;
  const FMCA::Index K = 5;
  const FMCA::Index mpole_deg = 0;
  const FMCA::Scalar ridge_parameter = 0 * npts;
  const FMCA::CovarianceKernel kernel("MaternNu", .1, 1., 0.5);
  FMCA::Matrix P = FMCA::Matrix::Random(dim, npts);
  P.row(0).setLinSpaced(npts, -1, 1);
  T.tic();
  const Moments mom(P, mpole_deg);
  const MatrixEvaluator mat_eval(mom, kernel);
  H2ClusterTree CT(mom, 20, P);
  T.toc("cluster tree:                ");
  std::vector<FMCA::Index> inv_idcs(P.cols());
  for (FMCA::Index i = 0; i < inv_idcs.size(); ++i) {
    inv_idcs[CT.indices()[i]] = i;
    inv_idcs[i] = i;
  }
  T.tic();
  const FMCA::Vector mdv = FMCA::minDistanceVector(CT, P);
  T.toc("min dist:                    ");
  const FMCA::Scalar fill_distance = mdv.maxCoeff();
  const FMCA::Scalar search_radius =
      K * fill_distance * std::abs(std::log(fill_distance));
  std::cout << "fill distance:                " << fill_distance << std::endl;
  std::cout << "search radius:                " << search_radius << std::endl;
  std::cout << "ridge parameter:              " << ridge_parameter << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  std::vector<std::vector<FMCA::Index>> epsnn(P.cols());
  T.tic();
#pragma omp parallel for
  for (FMCA::Index i = 0; i < epsnn.size(); ++i) {
    epsnn[i] = FMCA::epsNN(CT, P, P.col(i), search_radius);
  }
  T.toc("epsnn:                       ");
  T.tic();
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
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
    Kloc.diagonal().array() += ridge_parameter;
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
  FMCA::IO::print2ascii("P.txt", P);
  FMCA::IO::print2spascii("invKLag.txt", invK, "w");
  {
    FMCA::Matrix X(npts, 100), Y1(npts, 100), Y2(npts, 100);
    X.setZero();
    for (auto i = 0; i < 100; ++i) {
      FMCA::Index index = rand() % P.cols();
      FMCA::Vector col = kernel.eval(P, P.col(CT.indices()[index]));
      Y1.col(i) =
          col(Eigen::Map<const FMCA::iVector>(CT.indices(), CT.block_size()));
      X(index, i) = 1;
    }
    Y2 = invK * (Y1 + ridge_parameter * X);
    FMCA::Scalar err = (Y2 - X).norm() / X.norm();
    std::cout << "inverse error:                " << err << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  T.tic();
  const HMatrix hmat(CT, mat_eval, 0.8, 1e-6);
  T.toc("elapsed time H-matrix:       ");
  hmat.statistics();
  {
    FMCA::Matrix X(npts, 100), Y1(npts, 100), Y2(npts, 100);
    X.setZero();
    for (auto i = 0; i < 100; ++i) {
      FMCA::Index index = rand() % P.cols();
      FMCA::Vector col = kernel.eval(P, P.col(CT.indices()[index]));
      Y1.col(i) =
          col(Eigen::Map<const FMCA::iVector>(CT.indices(), CT.block_size()));
      X(index, i) = 1;
    }
    Y2 = hmat * X;
    FMCA::Scalar err = (Y1 - Y2).norm() / Y1.norm();
    std::cout << "compression error Hmatrix:    " << err << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Matrix X(npts, 100), Y1(npts, 100), Y2(npts, 100);
  X.setRandom();
  Y1 = hmat * X + ridge_parameter * X;
  Y2 = invK * Y1;
  std::cout << "inverse error vs H-Matrix:    " << (Y2 - X).norm() / X.norm()
            << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  FMCA::Vector rhs(npts);
  FMCA::Vector x(npts);
  FMCA::Vector y(npts);
  rhs.setOnes();
  x.setZero();
  FMCA::Vector res = rhs;
  FMCA::Scalar err = 10;
  FMCA::Index iter = 0;
  while (err > 1e-10 && iter < 1e4) {
    x += .5 * invK * res;
    y = hmat * x + ridge_parameter * x;
    res = rhs - y;
    err = res.norm() / rhs.norm();
    ++iter;
  }
  std::cout << "Richardson error: " << err << " iterations: " << iter
            << std::endl;
  return 0;
}

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
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                         Eigen::MetisOrdering<int>>;

#include "../FMCA/Clustering"
#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/src/Clustering/epsNN.h"
#include "../FMCA/src/H2Matrix/H2MatrixProjector.h"
#include "../FMCA/src/util/Tictoc.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 2000;
  const FMCA::Index dim = 2;
  const FMCA::Index K = 2;
  const FMCA::Index mpole_deg = 5;
  const FMCA::Scalar eta = 1. / dim;
  const FMCA::CovarianceKernel kernel("EXPONENTIAL", .1);
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
  triplets.clear();
  T.tic();
#pragma omp parallel for
  for (FMCA::Index i = 0; i < epsnn.size(); ++i) {
    const FMCA::Index locN = epsnn[i].size();
    std::vector<Eigen::Triplet<FMCA::Scalar>> local_triplets;
    for (FMCA::Index j = 0; j < locN; ++j) {
      const FMCA::Index row_id = inv_idcs[i];
      const FMCA::Index col_id = inv_idcs[epsnn[i][j]];
      const FMCA::Scalar val = kernel.eval(P.col(i), P.col(epsnn[i][j]))(0, 0);
      if (row_id <= col_id)
        local_triplets.push_back(
            Eigen::Triplet<FMCA::Scalar>(row_id, col_id, val));
    }
#pragma omp critical
    triplets.insert(triplets.end(), local_triplets.begin(),
                    local_triplets.end());
  }
  T.toc("local kernel triplets:       ");
  Eigen::SparseMatrix<FMCA::Scalar> Ksparse(npts, npts);
  Ksparse.setFromTriplets(triplets.begin(), triplets.end());
  std::cout << "anz:                          " << triplets.size() / npts
            << std::endl;
  T.tic();
  EigenCholesky solver;
  std::cout << Ksparse.norm() << std::endl;
  std::cout << Ksparse.diagonal().minCoeff() << std::endl;
  solver.compute(Ksparse);
   T.toc("time factorization: ");
  std::cout << "sinfo: " << (solver.info() == Eigen::Success) << std::endl;
  std::cout << "nz(L): "
            << solver.matrixL().nestedExpression().nonZeros() / P.cols()
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
  T.tic();
  FMCA::Matrix H2K = hmat.full();
  FMCA::Matrix invH2K = H2K.inverse();
  FMCA::H2MatrixProjector<H2Matrix> proj(hmat);
  H2Matrix invhmat = proj.project(invH2K);
  H2Matrix test = proj.project(H2K);

  T.toc("inverted full H2 matrix and projected:");
  FMCA::Matrix X(npts, 10), Y1(npts, 10), Y2(npts, 10);
  X.setRandom();
  Y1 = Ksparse * X;
  Y2 = invK * Y1;
  std::cout << "inverse error vs H2-Matrix:   " << (Y2 - X).norm() / X.norm()
            << std::endl;
  X.setRandom();
  Y1 = hmat * X;
  Y2 = invhmat * Y1;
  std::cout << "inverse H2-error:             " << (Y2 - X).norm() / X.norm()
            << std::endl;
  X.setRandom();
  Y1 = hmat * X;
  Y2 = test * X;
  std::cout << "H2-projector error:           " << (Y2 - Y1).norm() / Y1.norm()
            << std::endl;
  return 0;
}

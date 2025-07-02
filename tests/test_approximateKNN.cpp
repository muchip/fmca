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
#include <Eigen/Sparse>
#include <iostream>
#include <unordered_set>

#include "../FMCA/Clustering"
#include "../FMCA/src/Clustering/approximateKNN.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 6

int main(int argc, char *argv[]) {
  constexpr FMCA::Index npts = 100000;
  FMCA::Tictoc T;
  FMCA::Matrix P(DIM, npts);
  P.setRandom();
  FMCA::RandomProjectionTree CT(P, 100);
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> knn1 = FMCA::symKNN(CT, P, 10);
  T.toc("exact kNN:");
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> knn2 =
      FMCA::approximateSymKNN(P, 10, 100, 10);
  T.toc("approximate kNN:");
  Eigen::SparseMatrix<FMCA::Scalar, Eigen::RowMajor> A1(npts, npts);
  Eigen::SparseMatrix<FMCA::Scalar, Eigen::RowMajor> A2(npts, npts);
  A1.setFromTriplets(knn1.begin(), knn1.end());
  A2.setFromTriplets(knn2.begin(), knn2.end());
  std::cout << A1.nonZeros() << " " << A2.nonZeros() << std::endl;
  Eigen::SparseMatrix<FMCA::Scalar, Eigen::RowMajor> A1T = A1.transpose();
  Eigen::SparseMatrix<FMCA::Scalar, Eigen::RowMajor> A2T = A2.transpose();
  std::cout << (A1 - A1T).norm() / A1.norm() << std::endl;
  std::cout << (A2 - A2T).norm() / A2.norm() << std::endl;
  // compute recall
  size_t total = 0, found = 0;
  for (FMCA::Index i = 0; i < npts; ++i) {
    std::unordered_set<FMCA::Index> exact_neighbors;
    for (Eigen::SparseMatrix<FMCA::Scalar, Eigen::RowMajor>::InnerIterator it(
             A1, i);
         it; ++it)
      exact_neighbors.insert(it.col());
    std::unordered_set<FMCA::Index> approx_neighbors;
    for (Eigen::SparseMatrix<FMCA::Scalar, Eigen::RowMajor>::InnerIterator it(
             A2, i);
         it; ++it)
      approx_neighbors.insert(it.col());
    for (auto idx : exact_neighbors) {
      ++total;
      if (approx_neighbors.count(idx)) ++found;
    }
  }
  std::cout << found << " " << total << std::endl;
  std::cout << "Recall: " << (double)found / total << std::endl;
  return 0;
}

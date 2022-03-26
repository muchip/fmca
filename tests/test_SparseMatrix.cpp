// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include "../FMCA/src/util/SparseMatrix.h"
#include "../FMCA/src/util/Tictoc.h"

int main(int argc, char *argv[]) {
  const unsigned int npts = 2000;
  const Eigen::MatrixXd M = Eigen::MatrixXd::Random(npts, npts);
  Eigen::MatrixXd Sfull;
  FMCA::Tictoc T;
  FMCA::SparseMatrix<double> S(npts, npts);
  S.setZero();
  T.tic();
  S = FMCA::SparseMatrix<double>(M);
  T.toc("set matrix time: ");
  std::cout << "error: " << (S.full() - M).norm() << std::endl;
  T.tic();
  S.symmetrize();
  T.toc("symmetrize time: ");
  Sfull = S.full();
  std::cout << "error: " << (Sfull - Sfull.transpose()).norm() << std::endl;
  S = FMCA::SparseMatrix<double>(M);
  T.tic();
  S = FMCA::SparseMatrix<double>(S * M);
  T.toc("multiplication time");
  std::cout << "error: " << (S.full() - M * M).norm() << std::endl;
  S = FMCA::SparseMatrix<double>(M);
  T.tic();
  S = S + S;
  T.toc("addition time");
  std::cout << "error: " << (S.full() - 2 * M).norm() << std::endl;
  S = FMCA::SparseMatrix<double>(M);
  T.tic();
  S.transpose();
  T.toc("transposition time");
  std::cout << "error: " << (S.full() - M.transpose()).norm() << std::endl;

  S.setZero();
  for (auto i = 0; i < 500; ++i)
    for (auto j = 0; j < 500; ++j)
      S(rand() % npts, rand() % npts) = double(rand()) / double(RAND_MAX);
  std::cout << S.nnz() << std::endl;
  Sfull = S.full();
  T.tic();
  S = S.formatted_mult(S);
  T.toc("sparse multiplication time");
  std::cout << "error: " << (S.full() - Sfull * Sfull.transpose()).norm()
            << std::endl;

  return 0;
}

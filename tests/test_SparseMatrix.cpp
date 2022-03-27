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
  FMCA::Tictoc T;
  // small matrix tests
  {
    const unsigned int npts = 100;
    const Eigen::MatrixXd M = Eigen::MatrixXd::Random(npts, npts);
    Eigen::MatrixXd Sfull;
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
    T.toc("multiplication time full");
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
  }
  // large matrix tests
  {
    const unsigned int npts = 1000000;
    FMCA::SparseMatrix<double> S(npts, npts);
    FMCA::SparseMatrix<double> S2(npts, npts);
    FMCA::SparseMatrix<double> Sf(npts, npts);
    S.setZero();
    for (auto i = 0; i < 4000; ++i)
      for (auto j = 0; j < 4000; ++j)
        S(rand() % npts, rand() % npts) = double(rand()) / double(RAND_MAX);
    S.symmetrize();
    std::cout << "nnz of sparse matrix: " << S.nnz() << std::endl;
    Eigen::SparseMatrix<double> eigenS(npts, npts);
    Eigen::SparseMatrix<double> eigenT(npts, npts);
    auto trips = S.toTriplets();
    eigenS.setFromTriplets(trips.begin(), trips.end());
    eigenT = eigenS.transpose();
    std::cout << "sym error: " << (eigenS - eigenT).norm() << std::endl;
    T.tic();
    S2 = S * S;
    T.toc("sparse multiplication time");
    T.tic();
    eigenT = eigenS * eigenS;
    T.toc("Eigen::sparse multiplication time");
    trips = S2.toTriplets();
    eigenS.setFromTriplets(trips.begin(), trips.end());
    std::cout << "error: " << (eigenS - eigenT).norm() / eigenT.norm()
              << std::endl;
    T.tic();
    Sf = FMCA::SparseMatrix<double>::formatted_AtBT(S2, S, S);
    T.toc("sparse multiplication time formatted");
    trips = Sf.toTriplets();
    eigenS.setFromTriplets(trips.begin(), trips.end());
    std::cout << "error: " << (eigenS - eigenT).norm() / eigenT.norm()
              << std::endl;
    S.setZero();
    for (auto i = 0; i < 4000; ++i)
      for (auto j = 0; j < 4000; ++j)
        S(rand() % npts, rand() % npts) = double(rand()) / double(RAND_MAX);
    trips = S.toTriplets();
    eigenS.setFromTriplets(trips.begin(), trips.end());
    T.tic();
    S2 = S.transpose();
    T.toc("transposition: ");
    trips = S2.toTriplets();
    eigenT.setFromTriplets(trips.begin(), trips.end());
    eigenS = eigenS.transpose();
    std::cout << "error: " << (eigenS - eigenT).norm() / eigenS.norm()
              << std::endl;
    T.tic();
    S2 = S + S;
    T.toc("addition: ");
    trips = S2.toTriplets();
    eigenT.setFromTriplets(trips.begin(), trips.end());
    std::cout << "error: " << (2 * eigenS - eigenT).norm() / eigenS.norm()
              << std::endl;
  }
  return 0;
}

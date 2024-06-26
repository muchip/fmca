// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
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

#include "../FMCA/Clustering"
#include "../FMCA/CovarianceKernel"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 100000;
  const FMCA::Index dim = 3;
  const FMCA::Index K = 1;
  const FMCA::Scalar ridge_parameter = 0 * npts;
  const FMCA::CovarianceKernel kernel("MaternNu", .1, 1., 0.5);
  FMCA::Matrix P = FMCA::Matrix::Random(dim, npts);
  std::cout << std::string(72, '-') << std::endl;
  T.tic();
  const FMCA::ClusterTree CT(P, 10);
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.toc("cluster tree:                ");
  T.tic();
  Eigen::SparseMatrix<FMCA::Scalar> invK =
      FMCA::sparseKernelMatrixInverse(kernel, CT, P, K, ridge_parameter);
  T.toc("sparse inverse:              ");
  std::cout << "anz:                          " << invK.nonZeros() / npts
            << std::endl;
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
    std::cout << std::string(72, '-') << std::endl;
  }
  FMCA::IO::print2spascii("invK.txt", invK, "w");
  return 0;
}

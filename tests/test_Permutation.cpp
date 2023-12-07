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
#include <iostream>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/permutation.h"

#define DIM 10
#define NPTS 100000

int main() {
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  FMCA::ClusterTree CT(P, 10);
  FMCA::Vector test(NPTS);
  test.setRandom();
  auto Pmat = FMCA::permutationMatrix(CT);
  auto Pvec = FMCA::permutationVector(CT);
  const FMCA::Matrix PP = P * Pmat;
  for (FMCA::Index i = 0; i < P.cols(); ++i) {
    assert((PP.col(i) - P.col(CT.indices()[i])).norm() == 0 &&
           "permutation matrix error");
    assert((P.col(Pvec(i)) - P.col(CT.indices()[i])).norm() == 0 &&
           "permutation vector error");
  }
  assert((test(Pvec) - Pmat.transpose() * test).norm() / test.norm() <
             FMCA_ZERO_TOLERANCE &&
         "permutation transpose error");
  return 0;
}

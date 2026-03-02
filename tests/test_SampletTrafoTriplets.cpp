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

#include "../FMCA/Samplets"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 3
#define NPTS 1000000

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  const FMCA::Index dtilde = 4;
  std::cout << "dtilde:                       " << dtilde << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  const SampletTree st(samp_mom, 0, P);
  // T.tic();
  // auto trips = st.transformationMatrixTriplets();
  // T.toc("old trips: ");
  // FMCA::SparseMatrix S(NPTS, NPTS);
  // S.setFromTriplets(trips.begin(), trips.end());
  T.tic();
  auto trips2 = st.transformationMatrixTriplets2();
  T.toc("new trips: ");
  FMCA::SparseMatrix S2(NPTS, NPTS);
  S2.setFromTriplets(trips2.begin(), trips2.end());

  std::cout << (S2).norm() << std::endl << std::endl;

  return 0;
}

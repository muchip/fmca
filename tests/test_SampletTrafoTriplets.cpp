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

#define DIM 2
#define NPTS 15

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  const FMCA::Index dtilde = 1;
  std::cout << "dtilde:                       " << dtilde << std::endl;
  const SampletMoments samp_mom(P, dtilde - 1);
  const SampletTree st(samp_mom, 0, P);
  auto trips = st.transformationMatrixTriplets();
  FMCA::SparseMatrix S(NPTS, NPTS);
  S.setFromTriplets(trips.begin(), trips.end());
  auto trips2 = st.transformationMatrixTriplets2();
  FMCA::SparseMatrix S2(NPTS, NPTS);
  S2.setFromTriplets(trips2.begin(), trips2.end());

  std::cout << FMCA::Matrix(S) << std::endl << std::endl;
  std::cout << FMCA::Matrix(S2) << std::endl << std::endl;

  return 0;
}

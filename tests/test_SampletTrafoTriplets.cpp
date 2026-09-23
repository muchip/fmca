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
#include <FMCA/src/util/Tictoc.h>

#include <Eigen/Dense>
#include <FMCA/Samplets>
#include <iostream>

#define DIM 3
#define NPTS 10000

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
  T.tic();
  auto trips2 = st.transformationMatrixTriplets2();
  T.toc("new trips: ");
  FMCA::SparseMatrix S2(NPTS, NPTS);
  S2.setFromTriplets(trips2.begin(), trips2.end());

  std::cout << (S2).norm() << std::endl << std::endl;
  FMCA::SparseMatrix I(S2.cols(), S2.cols());
  I.setIdentity();
  std::cout << "orthogonality error: "
            << (S2.transpose() * S2 - I).norm() / std::sqrt(S2.rows()) << " / "
            << (S2 * S2.transpose() - I).norm() / std::sqrt(S2.rows())
            << std::endl;
  return 0;
}

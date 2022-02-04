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
//
#define FMCA_CLUSTERSET_
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Samplets"
#include "TestParameters.h"

int main() {
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  const double threshold = 1e-15;
  const FMCA::NystromMatrixEvaluator<FMCA::H2SampletTree, exponentialKernel>
      nm_eval(P, function);

  for (int dtilde = 1; dtilde <= 4; ++dtilde) {
    for (double eta = 1; eta >= 0; eta -= 0.5) {
      std::cout << "dtilde= " << dtilde << " eta= " << eta << std::endl;
      FMCA::H2SampletTree ST(P, LEAFSIZE, dtilde, MPOLE_DEG);
      FMCA::symmetric_compressor_impl<FMCA::H2SampletTree> Scomp;
      Scomp.compress(ST, nm_eval, eta, threshold);
      const auto &trips = Scomp.pattern_triplets();
      Eigen::MatrixXd K;
      nm_eval.compute_dense_block(ST, ST, &K);
      ST.sampletTransformMatrix(K);
      Eigen::SparseMatrix<double> S(K.rows(), K.cols());
      S.setFromTriplets(trips.begin(), trips.end());
      std::cout << Eigen::MatrixXd(S) << std::endl;
      //S += S.triangularView<Eigen::StrictlyUpper>().transpose();
      std::cout << "compression error: " << (S - K).norm() / K.norm()
                << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }
  }
  return 0;
}

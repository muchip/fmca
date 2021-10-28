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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Samplets"
#include "TestParameters.h"

int main() {
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);

  for (double dtilde = 1; dtilde <= 10; ++dtilde) {
    std::cout << "dtilde= " << dtilde << std::endl;
    const FMCA::SampletTreeQR ST(P, LEAFSIZE, dtilde);
    FMCA::SampleMomentComputer<FMCA::SampletTreeQR,
                               FMCA::MultiIndexSet<FMCA::TotalDegree>>
        mom_comp;
    mom_comp.init(P.rows(), dtilde);
    Eigen::MatrixXd Pol = mom_comp.moment_matrix(P, ST);
    Pol = Pol.topRows(mom_comp.mdtilde());
    double err = 0;
    Pol = ST.sampletTransform(Pol.transpose());
    err = Pol.bottomRows(Pol.cols() - ST.nscalfs()).colwise().norm().sum();
    std::cout << "average vanishing error: " << err / Pol.rows() << std::endl;
    Eigen::MatrixXd Q =
        ST.sampletTransform(Eigen::MatrixXd::Identity(P.cols(), P.cols()));
    std::cout << "basis orthogonality error: "
              << (Q.transpose() * Q -
                  Eigen::MatrixXd::Identity(P.cols(), P.cols()))
                         .norm() /
                     sqrt(P.cols())
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }

  return 0;
}

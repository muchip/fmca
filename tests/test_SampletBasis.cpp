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

using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);

  for (double dtilde = 1; dtilde <= 10; ++dtilde) {
    std::cout << "dtilde= " << dtilde << std::endl;
    const SampletMoments samp_mom(P, dtilde - 1);
    const SampletTree st(samp_mom, 0, P);

    Eigen::MatrixXd Pol = samp_mom.moment_matrix(st);
    Pol = Pol.topRows(samp_mom.mdtilde());
    double err = 0;
    Pol = st.sampletTransform(Pol.transpose());
    err = Pol.bottomRows(Pol.cols() - st.nscalfs()).colwise().norm().sum();
    std::cout << "average vanishing error: " << err / Pol.rows() << std::endl;
    Eigen::MatrixXd Q =
        st.sampletTransform(Eigen::MatrixXd::Identity(P.cols(), P.cols()));
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

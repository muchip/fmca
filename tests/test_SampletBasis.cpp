// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU General Public License version 3
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Samplets"

#define DIM 2
#define NPTS 1000

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);

  for (FMCA::Scalar dtilde = 1; dtilde <= 10; ++dtilde) {
    std::cout << "dtilde:                       " << dtilde << std::endl;
    const SampletMoments samp_mom(P, dtilde - 1);
    const SampletTree st(samp_mom, 0, P);

    FMCA::Matrix Pol = samp_mom.moment_matrix(st);
    Pol = Pol.topRows(samp_mom.mdtilde());
    FMCA::Scalar err = 0;
    Pol = st.sampletTransform(Pol.transpose());
    err = Pol.bottomRows(Pol.cols() - st.nscalfs()).colwise().norm().sum();
    std::cout << "average vanishing error:      " << err / Pol.rows()
              << std::endl;
    FMCA::Matrix Q =
        st.sampletTransform(Eigen::MatrixXd::Identity(P.cols(), P.cols()));
    std::cout << "basis orthogonality error:    "
              << (Q.transpose() * Q -
                  Eigen::MatrixXd::Identity(P.cols(), P.cols()))
                         .norm() /
                     sqrt(P.cols())
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }

  return 0;
}

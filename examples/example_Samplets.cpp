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

#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/src/util/Errors.h>
#include <FMCA/src/util/Tictoc.h>

using Interpolator = FMCA::TotalDegreeInterpolator<FMCA::FloatType>;
using SampletInterpolator = FMCA::MonomialInterpolator<FMCA::FloatType>;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const unsigned int mp_deg = 3;
  const unsigned int dim = atoi(argv[1]);
  FMCA::Tictoc T;
  std::cout << std::string(60, '-') << std::endl;
  for (auto i = 2; i < 7; ++i) {
    const unsigned int npts = std::pow(10, i);
    const Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
    T.tic();
    const SampletMoments samp_mom(P, mp_deg);
    const SampletTree st(samp_mom, 0, P);
    T.toc("SampletTree setup: ");
    T.tic();
    const Moments mom(P, mp_deg);
    const H2SampletTree st2(mom, samp_mom, 0, P);
    T.toc("H2SampletTree setup: ");

    std::cout << std::string(60, '-') << std::endl;
  }

  return 0;
}

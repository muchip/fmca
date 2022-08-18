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
#include "../FMCA/src/Samplets/samplet_transformer.h"
#define DIM 2
#define NPTS 100000

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;

int main() {
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  const FMCA::Index dtilde = 3;
  const SampletMoments samp_mom(P, dtilde - 1);
  const SampletTree st(samp_mom, 0, P);
  FMCA::Matrix X(NPTS, 100);
  X.setRandom();
  auto Yref = st.sampletTransform(X);
  {
    FMCA::SampletTransformer<SampletTree> s_transform(st, 0);
    auto Y = s_transform.transform(X);
    assert((Yref - Y).norm() / Yref.norm() < FMCA_ZERO_TOLERANCE &&
           "error in samplet transformer for min_level = 0");
  }
  for (auto lvl = 1; lvl < 10; ++lvl) {
    FMCA::SampletTransformer<SampletTree> s_transform(st, lvl);
    auto TtTX = s_transform.inverseTransform(s_transform.transform(X));
    assert((X - TtTX).norm() / X.norm() < 10 * FMCA_ZERO_TOLERANCE &&
           "error in samplet transformer for min_level > 0");
  }
  return 0;
}

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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using KDTree = FMCA::SampletTree<FMCA::KDTree>;
// using KDTree = FMCA::KDTree;

int main() {
  constexpr FMCA::Index l = 11;
  constexpr FMCA::Index d = 2;
  constexpr FMCA::Index N = 1 << l;
  constexpr FMCA::Scalar h = 1. / N;
  constexpr FMCA::Index Nd = std::pow(N, d);
  constexpr FMCA::Index dtilde = 4;
  FMCA::Tictoc T;
  FMCA::Matrix P(d, Nd);
  FMCA::Vector data(Nd);
  T.tic();
  {
    // generate a uniform grid
    FMCA::Vector pt(d);
    pt.setZero();
    FMCA::Index p = 0;
    FMCA::Index i = 0;
    while (pt(d - 1) < N) {
      if (pt(p) >= N) {
        pt(p) = 0;
        ++p;
      } else {
        P.col(i++) = h * (pt.array() + 0.5).matrix();
        p = 0;
      }
      pt(p) += 1;
    }
  }
  {
    FMCA::Vector pt(d);
    // create a non axis aligned jump
    pt.setOnes();
    pt /= std::sqrt(d);
    for (FMCA::Index i = 0; i < Nd; ++i)
      data(i) = P.col(i).dot(pt) > 0.5 * sqrt(d);
  }
  T.toc("data generation: ");
  std::cout << "Nd=" << Nd << std::endl;
  //////////////////////////////////////////////////////////////////////
  const SampletMoments samp_mom(P, dtilde - 1);
  SampletTree st(samp_mom, 0, P);
  KDTree kdst(samp_mom, 0, P);

  const FMCA::Vector scoeffs = st.sampletTransform(st.toClusterOrder(data));
  std::vector<FMCA::Index> levels = FMCA::internal::sampletLevelMapper(st);
  FMCA::Index max_l = 0;
  for (const auto &it : levels) max_l = max_l < it ? it : max_l;
  std::vector<FMCA::Scalar> max_c(max_l + 1);
  for (FMCA::Index i = 0; i < scoeffs.size(); ++i)
    max_c[levels[i]] = max_c[levels[i]] < std::abs(scoeffs[i])
                           ? std::abs(scoeffs[i])
                           : max_c[levels[i]];
  for (FMCA::Index i = 1; i < max_c.size(); ++i)
    std::cout << "c=" << max_c[i] / std::sqrt(Nd)
              << " alpha=" << std::log(max_c[i - 1] / max_c[i]) / (std::log(2))
              << std::endl;
  std::cout << "---------------------------" << std::endl;
  const FMCA::Vector scoeffs_kd = kdst.sampletTransform(kdst.toClusterOrder(data));
  std::vector<FMCA::Index> levels_kd = FMCA::internal::sampletLevelMapper(kdst);
  FMCA::Index max_l_kd = 0;
  for (const auto &it : levels_kd) max_l_kd = max_l_kd < it ? it : max_l_kd;
  std::vector<FMCA::Scalar> max_c_kd(max_l_kd + 1);
  for (FMCA::Index i = 0; i < scoeffs_kd.size(); ++i)
    max_c_kd[levels_kd[i]] = max_c_kd[levels_kd[i]] < std::abs(scoeffs_kd[i])
                           ? std::abs(scoeffs_kd[i])
                           : max_c_kd[levels_kd[i]];
  for (FMCA::Index i = 1; i < max_c_kd.size(); ++i)
    std::cout << "c=" << max_c_kd[i] / std::sqrt(Nd)
              << " alpha=" << std::log(max_c_kd[i - 1] / max_c_kd[i]) / (std::log(2))
              << std::endl;
  return 0;
}

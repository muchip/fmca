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
#ifndef FMCA_MATRIXEVALUATORS_GALERKINHELPER_H_
#define FMCA_MATRIXEVALUATORS_GALERKINHELPER_H_

namespace FMCA {
bool is_admissble(const Eigen::Matrix3d &T1, const Eigen::Matrix3d &T2) {
  Eigen::Vector3d mp1 = T1.col(0) + 1. / 3. * (T1.col(1) + T1.col(2));
  Eigen::Vector3d mp2 = T2.col(0) + 1. / 3. * (T2.col(1) + T2.col(2));
}
} // namespace FMCA
#endif

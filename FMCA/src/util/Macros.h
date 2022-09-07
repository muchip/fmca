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
#ifndef FMCA_UTIL_MACROS_H_
#define FMCA_UTIL_MACROS_H_

namespace FMCA {
#ifndef M_PI
#define FMCA_PI 3.14159265358979323846264338327950288
#else
#define FMCA_PI M_PI
#endif
#define FMCA_BBOX_THREASHOLD 1e-2
#define FMCA_ZERO_TOLERANCE 2e-16

#define FMCA_INF  std::numeric_limits<double>::infinity()
typedef unsigned int Index;

typedef double Scalar;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1u> Vector;

typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> iMatrix;

typedef Eigen::Matrix<Index, Eigen::Dynamic, 1u> iVector;
} // namespace FMCA

#endif

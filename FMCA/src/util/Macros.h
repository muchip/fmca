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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>

namespace FMCA {
#ifndef M_PI
#define FMCA_PI 3.14159265358979323846264338327950288
#else
#define FMCA_PI M_PI
#endif

// define primitive types used throughout the toolbox
#define FMCA_INDEX unsigned int
#define FMCA_SCALAR double
#define FMCA_INF std::numeric_limits<FMCA_SCALAR>::infinity()
#define FMCA_ZERO_TOLERANCE std::numeric_limits<FMCA_SCALAR>::epsilon()
#define FMCA_BBOX_THREASHOLD FMCA_ZERO_TOLERANCE

typedef FMCA_INDEX Index;

typedef FMCA_SCALAR Scalar;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1u> Vector;

typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> iMatrix;

typedef Eigen::Matrix<Index, Eigen::Dynamic, 1u> iVector;

template <typename T>
using Triplet = Eigen::Triplet<T>;

typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>
    SparseMatrix;
}  // namespace FMCA

#endif

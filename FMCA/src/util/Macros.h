// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_MACROS_H_
#define FMCA_UTIL_MACROS_H_

namespace FMCA {
#ifndef M_PI
#define FMCA_PI 3.14159265358979323846264338327950288
#else
#define FMCA_PI M_PI
#endif
#define FMCA_BBOX_THREASHOLD 5e-2
#define FMCA_ZERO_TOLERANCE 1e-14

typedef unsigned int IndexType;
typedef double FloatType;
} // namespace FMCA

#endif

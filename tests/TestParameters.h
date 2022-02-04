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
#ifndef FMCA_TESTS_TESTPARAMETERS_H_
#define FMCA_TESTS_TESTPARAMETERS_H_

#define NPTS 10
#define DIM 2
#define MPOLE_DEG 3
#define LEAFSIZE 10

struct exponentialKernel {
  double operator()(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y) const {
    return exp(-10 * (x - y).norm());
  }
};

#endif

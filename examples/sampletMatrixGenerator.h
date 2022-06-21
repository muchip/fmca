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

#ifndef FMCA_SAMPLETMATRIXGENERATOR_
#define FMCA_SAMPLETMATRIXGENERATOR_

#include <vector>
struct CRSmatrix {
  std::vector<int> ia;
  std::vector<int> ja;
  std::vector<double> a;
};

CRSmatrix sampletMatrixGenerator(const Eigen::MatrixXd &P,
                                 const unsigned int mp_deg,
                                 const unsigned int dtilde, const double eta,
                                 double threshold, const double ridgep);

#endif

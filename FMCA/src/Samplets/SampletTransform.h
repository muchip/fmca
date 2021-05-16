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
#ifndef FMCA_SAMPLETS_SAMPLETTRANSFORM_H_
#define FMCA_SAMPLETS_SAMPLETTRANSFORM_H_

template <typename SampletTree, typename fhandle>
Eigen::Matrix<typename SampletTree::value_type, Eigen::Dynamic, 1>
sampletTransform(const Eigen::Matrix<typename SampletTree::value_type,
                                     SampletTree::dimension, Eigen::Dynamic> &P,
                 const SampletTree &ST, const fhandle &fun) {
  Eigen::Matrix<typename SampletTree::value_type, Eigen::Dynamic, 1> retval(
      P.cols());
}

#endif

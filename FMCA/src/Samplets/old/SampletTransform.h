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
namespace FMCA {
template <typename ClusterTree, typename fhandle>
Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic, 1>
functionEvaluator(
    const Eigen::Matrix<typename ClusterTree::value_type,
                        ClusterTree::dimension, Eigen::Dynamic> &P,
    const ClusterTree &CT, const fhandle &fun) {
  Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic, 1> retval;
  auto idcs = CT.get_indices();
  retval.resize(idcs.size());
  for (auto i = 0; i < idcs.size(); ++i)
    retval(i) = fun(P.col(idcs[i]));
  return retval;
}

} // namespace FMCA
#endif

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
#ifndef FMCA_SAMPLETS_MOMENTCOMPUTER_H_
#define FMCA_SAMPLETS_MOMENTCOMPUTER_H_

namespace FMCA {
/**
 *  \brief computes the moments for a given cluster
 **/
template <typename ClusterTree>
Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic, Eigen::Dynamic>
momentComputer(const Eigen::Matrix<typename ClusterTree::value_type,
                                   ClusterTree::dimension, Eigen::Dynamic> &P,
               const ClusterTree &CT,
               const MultiIndexSet<ClusterTree::dimension> &idcs) {
  Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                Eigen::Dynamic>
      retval(idcs.get_MultiIndexSet().size(), CT.get_indices().size());
  Eigen::VectorXd mp = 0.5 * (CT.get_bb().col(0) + CT.get_bb().col(1));

  IndexType i = 0;
  retval.setOnes();
  for (auto j = 0; j < retval.cols(); ++j) {
    i = 0;
    for (auto it : idcs.get_MultiIndexSet()) {
      for (auto k = 0; k < ClusterTree::dimension; ++k)
        // make sure that 0^0 = 1
        if (it[k])
          retval(i, j) *= std::pow(P(k, CT.get_indices()[j]) - mp(k), it[k]);
      ++i;
    }
  }
  return retval;
}

/**
 *  \brief computes the transformation matrix from the son cluster moments
 *         to the dad cluster moments.
 **/
template <typename ClusterTree>
Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic, Eigen::Dynamic>
momentShifter(
    const Eigen::Matrix<typename ClusterTree::value_type,
                        ClusterTree::dimension, 1> &shift,
    const MultiIndexSet<ClusterTree::dimension> &idcs,
    const Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                        Eigen::Dynamic> &mult_coeffs) {
  Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                Eigen::Dynamic>
      retval = mult_coeffs;
  if (shift.norm() < FMCA_ZERO_TOLERANCE) {
    return retval;
  }
  IndexType i = 0;
  IndexType j = 0;
  typename ClusterTree::value_type weight;
  for (auto it1 : idcs.get_MultiIndexSet()) {
    j = 0;
    for (auto it2 : idcs.get_MultiIndexSet()) {
      // check if the multinomial coefficient is non-zero
      if (retval(j, i)) {
        for (auto k = 0; k < shift.size(); ++k)
          // make sure that 0^0 = 1
          if (it2[k] - it1[k])
            retval(j, i) *= std::pow(shift(k), it2[k] - it1[k]);
      }
      ++j;
    }
    ++i;
  }
  return retval;
}

} // namespace FMCA
#endif

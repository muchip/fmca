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
    for (auto it = idcs.get_MultiIndexSet().begin();
         it != idcs.get_MultiIndexSet().end(); ++it) {
      for (auto k = 0; k < ClusterTree::dimension; ++k)
        retval(i, j) *= std::pow(P(k, CT.get_indices()[j]) - mp(k), (*it)[k]);
      ++i;
    }
  }
  return retval;
}

template <typename ClusterTree, typename Derived1, typename Derived2>
Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic, Eigen::Dynamic>
momentShifter(const Eigen::MatrixBase<Derived1> &Mom,
              const Eigen::Matrix<typename ClusterTree::value_type,
                                  ClusterTree::dimension, 1> &mp_dad,
              const Eigen::Matrix<typename ClusterTree::value_type,
                                  ClusterTree::dimension, 1> &mp_son,
              const MultiIndexSet<ClusterTree::dimension> &idcs,
              const Eigen::MatrixBase<Derived2> &mult_coeffs) {
  Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                Eigen::Dynamic>
      retval(Mom.rows(), Mom.cols());
  retval.setZero();
  Eigen::Matrix<typename ClusterTree::value_type, ClusterTree::dimension, 1>
      mp_shift = mp_son - mp_dad;
  if (mp_shift.norm() < FMCA_ZERO_TOLERANCE) {
    retval = Mom;
    return retval;
  }
  IndexType i = 0;
  IndexType j = 0;
  typename ClusterTree::value_type weight;
  typename ClusterTree::value_type exponent;
  typename ClusterTree::value_type base;
  retval.setZero();
  for (auto it1 : idcs.get_MultiIndexSet()) {
    j = 0;
    for (auto it2 : idcs.get_MultiIndexSet()) {
      // check if the multinomial coefficient is non-zero
      if (mult_coeffs(j, i)) {
        weight = mult_coeffs(j, i);
        for (auto k = 0; k < mp_shift.size(); ++k) {
          base = mp_shift(k);
          exponent = it1[k] - it2[k];
          if (abs(base) < FMCA_ZERO_TOLERANCE) {
            if (abs(exponent) > FMCA_ZERO_TOLERANCE) {
              weight = 0;
              break;
            }
          } else
            weight *= std::pow(base, exponent);
        }
        retval.row(i) += weight * Mom.row(j);
      }
      ++j;
      if (j > i)
        break;
    }
    ++i;
  }
  return retval;
} // namespace FMCA

} // namespace FMCA
#endif

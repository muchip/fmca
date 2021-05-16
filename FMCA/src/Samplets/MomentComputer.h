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

  unsigned int i = 0;
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
momentShifter(const Eigen::MatrixBase<Derived1> &Mom, const ClusterTree &CTdad,
              const ClusterTree &CTson,
              const MultiIndexSet<ClusterTree::dimension> &idcs,
              const Eigen::MatrixBase<Derived2> &mult_coeffs) {
  Eigen::Matrix<typename ClusterTree::value_type, Eigen::Dynamic,
                Eigen::Dynamic>
      retval(Mom.rows(), Mom.cols());
  retval.setZero();
  Eigen::VectorXd mp_dad =
      0.5 * (CTdad.get_bb().col(0) + CTdad.get_bb().col(1));
  Eigen::VectorXd mp_son =
      0.5 * (CTson.get_bb().col(0) + CTson.get_bb().col(1));

  unsigned int i = 0;
  retval.setOnes();
  for (auto j = 0; j < retval.rows(); ++j) {
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

} // namespace FMCA
#endif

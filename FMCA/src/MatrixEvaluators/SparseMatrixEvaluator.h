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
#ifndef FMCA_MATRIXEVALUATORS_SPARSEMATRIXEVALUATOR_H_
#define FMCA_MATRIXEVALUATORS_SPARSEMATRIXEVALUATOR_H_

#include "../util/SparseMatrix.h"
namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief provides the required two methods interpolate_kernel and
 *         compute_dense_block to compute the compressed matrix of
 *         a given Nystrom matrix that is fully described by these
 *         two routines.
 **/
template <typename Scalar> struct SparseMatrixEvaluator {
  typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eigenVector;
  typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
      eigenMatrix;
  typedef Scalar value_type;

  SparseMatrixEvaluator(const SparseMatrix<Scalar> &M) : M_(M) {}
  /**
   *  \brief provides the kernel evaluation for the H2-matrix, in principle
   *         this method could also be used to return the desired block from
   *         a precomputed H2-matrix! The variant below computes entries
   *         on the fly
   **/
  template <typename Derived>
  void interpolate_kernel(const ClusterTreeBase<Derived> &TR,
                          const ClusterTreeBase<Derived> &TC,
                          eigenMatrix *mat) const {
    mat->resize(TR.derived().V().rows(), TC.derived().V().rows());
    mat->setZero();
    return;
  }
  /**
   *  \brief provides the evaluaton of a dense matrix block for a given
   *         cluster pair
   **/
  template <typename Derived>
  void compute_dense_block(const ClusterTreeBase<Derived> &TR,
                           const ClusterTreeBase<Derived> &TC,
                           eigenMatrix *retval) const {
    retval->resize(TR.indices().size(), TC.indices().size());
    for (auto j = 0; j < TC.indices().size(); ++j)
      for (auto i = 0; i < TR.indices().size(); ++i)
        (*retval)(i, j) = M_(TR.indices()[i], TC.indices()[j]);
    return;
  }

  const SparseMatrix<Scalar> &M_;
};

} // namespace FMCA
#endif

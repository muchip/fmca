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
struct SparseMatrixEvaluator {
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
                          Matrix *mat) const {
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
                           Matrix *retval) const {
    retval->resize(TR.block_size(), TC.block_size());
    for (auto j = 0; j < TC.block_size(); ++j)
      for (auto i = 0; i < TR.block_size(); ++i)
        (*retval)(i, j) = M_(TR.indices()[i], TC.indices()[j]);
    return;
  }

  const SparseMatrix<Scalar> &M_;
};

} // namespace FMCA
#endif
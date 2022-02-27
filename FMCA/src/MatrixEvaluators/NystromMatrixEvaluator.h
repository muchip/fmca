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
#ifndef FMCA_MATRIXEVALUATORS_NYSTROMMATRIXEVALUATOR_H_
#define FMCA_MATRIXEVALUATORS_NYSTROMMATRIXEVALUATOR_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief provides the required two methods interpolate_kernel and
 *         compute_dense_block to compute the compressed matrix of
 *         a given Nystrom matrix that is fully described by these
 *         two routines.
 **/
template <typename Moments, typename Kernel> struct NystromMatrixEvaluator {
  typedef typename Moments::eigenVector eigenVector;
  typedef typename Moments::eigenMatrix eigenMatrix;
  typedef typename eigenMatrix::Scalar value_type;

  NystromMatrixEvaluator(const Moments &mom) : mom_(mom) {}
  NystromMatrixEvaluator(const Moments &mom, const Kernel &kernel)
      : mom_(mom), kernel_(kernel) {}

  void init(const Kernel &kernel) {
    kernel_ = kernel;
    return;
  }
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
    eigenMatrix XiX = mom_.interp().Xi().cwiseProduct(TR.bb().col(2).replicate(
                          1, mom_.interp().Xi().cols())) +
                      TR.bb().col(0).replicate(1, mom_.interp().Xi().cols());
    eigenMatrix XiY = mom_.interp().Xi().cwiseProduct(TC.bb().col(2).replicate(
                          1, mom_.interp().Xi().cols())) +
                      TC.bb().col(0).replicate(1, mom_.interp().Xi().cols());
    mat->resize(XiX.cols(), XiX.cols());
    for (auto j = 0; j < mat->cols(); ++j)
      for (auto i = 0; i < mat->rows(); ++i)
        (*mat)(i, j) = kernel_(XiX.col(i), XiY.col(j));
    *mat = mom_.interp().invV() * (*mat) * mom_.interp().invV().transpose();
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
        (*retval)(i, j) = kernel_(mom_.P().col(TR.indices()[i]),
                                  mom_.P().col(TC.indices()[j]));
    return;
  }

  const Moments &mom_;
  Kernel kernel_;
};

} // namespace FMCA
#endif

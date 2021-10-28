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
#ifndef FMCA_SAMPLETS_NYSTROMMATRIXEVALUATOR_H_
#define FMCA_SAMPLETS_NYSTROMMATRIXEVALUATOR_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief provides the required two methods interpolate_kernel and
 *         compute_dense_block to compute the compressed matrix of
 *         a given Nystrom matrix that is fully described by these
 *         two routines.
 **/
template <typename Derived, typename Kernel>
struct NystromMatrixEvaluator {
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;

  NystromMatrixEvaluator(){};
  NystromMatrixEvaluator(const eigenMatrix &P, const Kernel &kernel) {
    init(P, kernel);
  };
  void init(const eigenMatrix &P, const Kernel &kernel) {
    P_ = std::addressof(P);
    kernel_ = kernel;
    return;
  }
  /**
   *  \brief provides the kernel evaluation for the H2-matrix, in principle
   *         this method could also be used to return the desired block from
   *         a precomputed H2-matrix! The variant below computes entries
   *         on the fly
   **/
  void interpolate_kernel(const Derived &TR, const Derived &TC,
                          eigenMatrix *retval) const {
    eigenMatrix XiX =
        TR.Xi().cwiseProduct(TR.bb().col(2).replicate(1, TR.Xi().cols())) +
        TR.bb().col(0).replicate(1, TR.Xi().cols());
    eigenMatrix XiY =
        TR.Xi().cwiseProduct(TC.bb().col(2).replicate(1, TR.Xi().cols())) +
        TC.bb().col(0).replicate(1, TR.Xi().cols());
    retval->resize(XiX.cols(), XiX.cols());
    for (auto j = 0; j < retval->cols(); ++j)
      for (auto i = 0; i < retval->rows(); ++i)
        (*retval)(i, j) = kernel_(XiX.col(i), XiY.col(j));
    return;
  }
  /**
   *  \brief provides the evaluaton of a dense matrix block for a given
   *         cluster pair
   **/
  void compute_dense_block(const Derived &TR, const Derived &TC,
                           eigenMatrix *retval) const {
    retval->resize(TR.indices().size(), TC.indices().size());
    for (auto j = 0; j < TC.indices().size(); ++j)
      for (auto i = 0; i < TR.indices().size(); ++i)
        (*retval)(i, j) =
            kernel_(P_->col(TR.indices()[i]), P_->col(TC.indices()[j]));
    return;
  }
  const eigenMatrix *P_;
  Kernel kernel_;
};

}  // namespace FMCA
#endif

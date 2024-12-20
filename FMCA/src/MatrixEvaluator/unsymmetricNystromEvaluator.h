// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_MATRIXEVALUATORS_UNSYMMETRICNYSTROMMATRIXEVALUATOR_H_
#define FMCA_MATRIXEVALUATORS_UNSYMMETRICNYSTROMMATRIXEVALUATOR_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief provides the required two methods interpolate_kernel and
 *         compute_dense_block to compute the compressed matrix of
 *         a given Nystrom matrix that is fully described by these
 *         two routines.
 **/
template <typename Moments, typename Kernel>
struct unsymmetricNystromEvaluator {
  unsymmetricNystromEvaluator(const Moments &r_mom, const Moments &c_mom)
      : r_mom_(r_mom), c_mom_(c_mom) {}
  unsymmetricNystromEvaluator(const Moments &r_mom, const Moments &c_mom,
                              const Kernel &kernel)
      : r_mom_(r_mom), c_mom_(c_mom), kernel_(kernel) {}

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
                          Matrix *mat) const {
    Matrix XiX = r_mom_.interp().Xi().cwiseProduct(
                     TR.bb().col(2).replicate(1, r_mom_.interp().Xi().cols())) +
                 TR.bb().col(0).replicate(1, r_mom_.interp().Xi().cols());
    Matrix XiY = c_mom_.interp().Xi().cwiseProduct(
                     TC.bb().col(2).replicate(1, c_mom_.interp().Xi().cols())) +
                 TC.bb().col(0).replicate(1, c_mom_.interp().Xi().cols());
    mat->resize(XiX.cols(), XiY.cols());
    for (auto j = 0; j < mat->cols(); ++j)
      for (auto i = 0; i < mat->rows(); ++i)
        (*mat)(i, j) = kernel_(XiX.col(i), XiY.col(j));
    *mat = r_mom_.interp().invV() * (*mat) * c_mom_.interp().invV().transpose();
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
        (*retval)(i, j) = kernel_(r_mom_.P().col(TR.indices()[i]),
                                  c_mom_.P().col(TC.indices()[j]));
    return;
  }

  /**
   *  \brief provides the evaluaton of a matrix entry given
   *         an index pair
   **/
  Scalar compute_entry(const Index i, const Index j) const {
    return kernel_(r_mom_.P().col(i), c_mom_.P().col(j));
  }

  /**
   *  \brief add an operator which makes it feel like a matrix
   **/
  Scalar operator()(const Index i, const Index j) const {
    return compute_entry(i, j);
  }

  const Moments &r_mom_;
  const Moments &c_mom_;
  Kernel kernel_;
};

}  // namespace FMCA
#endif

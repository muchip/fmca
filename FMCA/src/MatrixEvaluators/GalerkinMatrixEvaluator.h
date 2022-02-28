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
#ifndef FMCA_MATRIXEVALUATORS_GALERKINMATRIXEVALUATOR_H_
#define FMCA_MATRIXEVALUATORS_GALERKINMATRIXEVALUATOR_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief provides the required two methods interpolate_kernel and
 *         compute_dense_block to compute the compressed matrix of
 *         a given Nystrom matrix that is fully described by these
 *         two routines.
 **/
template <typename Moments> struct GalerkinMatrixEvaluatorSL {
  typedef typename Moments::eigenVector eigenVector;
  typedef typename Moments::eigenMatrix eigenMatrix;
  typedef typename eigenMatrix::Scalar value_type;
  const value_type cnst = 0.25 / FMCA_PI;
  GalerkinMatrixEvaluatorSL(const Moments &mom) : mom_(mom) {}
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
      for (auto i = 0; i < mat->rows(); ++i) {
        value_type r = (XiX.col(i) - XiY.col(j)).norm();
        (*mat)(i, j) = cnst / r;
      }
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
    for (auto j = 0; j < TC.indices().size(); ++j) {
      // set up first element
      Eigen::Matrix<value_type, 3, 3> element1;
      element1.col(0) = V_.row(F_(TC.indices()[j], 0)).transpose();
      element1.col(1) = V_.row(F_(TC.indices()[j], 1)).transpose();
      element1.col(2) = V_.row(F_(TC.indices()[j], 2)).transpose();
      element1.rightCols(2).colwise() -= element1.col(0);
      for (auto i = 0; i < TR.indices().size(); ++i) {
        // set up second element
        Eigen::Matrix<value_type, 3, 3> element2;
        element2.col(0) = V_.row(F_(TR.indices()[i], 0)).transpose();
        element2.col(1) = V_.row(F_(TR.indices()[i], 1)).transpose();
        element2.col(2) = V_.row(F_(TR.indices()[i], 2)).transpose();
        element2.rightCols(2).colwise() -= element2.col(0);

        (*retval)(i, j) = kernel_(mom_.P().col(TR.indices()[i]),
                                  mom_.P().col(TC.indices()[j]));
      }
    }
    return;
  }

  const Moments &mom_;
  const Quad::Quadrature<Quad::Midpoint> Rq_;
  const Quad::Quadrature<Quad::Radon> Rq_;
};

} // namespace FMCA
#endif

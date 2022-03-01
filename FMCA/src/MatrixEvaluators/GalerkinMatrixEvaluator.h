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
      element1.col(0) = mom_.V().row(mom_.F()(TC.indices()[j], 0)).transpose();
      element1.col(1) = mom_.V().row(mom_.F()(TC.indices()[j], 1)).transpose();
      element1.col(2) = mom_.V().row(mom_.F()(TC.indices()[j], 2)).transpose();
      element1.rightCols(2).colwise() -= element1.col(0);
      value_type area1 = 0.5 * (element1.col(1)).cross(element1.col(2)).norm();
      for (auto i = 0; i < TR.indices().size(); ++i) {
        // set up second element
        Eigen::Matrix<value_type, 3, 3> element2;
        element2.col(0) =
            mom_.V().row(mom_.F()(TR.indices()[i], 0)).transpose();
        element2.col(1) =
            mom_.V().row(mom_.F()(TR.indices()[i], 1)).transpose();
        element2.col(2) =
            mom_.V().row(mom_.F()(TR.indices()[i], 2)).transpose();
        element2.rightCols(2).colwise() -= element2.col(0);
        value_type area2 =
            0.5 * (element2.col(1)).cross(element2.col(2)).norm();
        // if two elements are not identical, we use a midpoint rule for
        // integration
        if (TR.indices()[i] != TC.indices()[j]) {
          value_type r =
              (element1.col(0) + element1.rightCols(2) * Mq_.xi.col(0) -
               element2.col(0) - element2.rightCols(2) * Mq_.xi.col(0))
                  .norm();
          (*retval)(i, j) =
              Mq_.w(0) * Mq_.w(0) * sqrt(area1) * sqrt(area2) * cnst / r;
        } else
          (*retval)(i, j) = cnst * sqrt(area1) * sqrt(area2);
      }
    }
    return;
  }

  const Moments &mom_;
  const Quad::Quadrature<Quad::Midpoint> Mq_;
  const Quad::Quadrature<Quad::Radon> Rq_;
};

} // namespace FMCA
#endif

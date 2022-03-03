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
#ifndef FMCA_MATRIXEVALUATORS_GALERKINMATRIXEVALUATORDL_H_
#define FMCA_MATRIXEVALUATORS_GALERKINMATRIXEVALUATORDL_H_

namespace FMCA {

/**
 *  \ingroup Samplets
 *  \brief provides the required two methods interpolate_kernel and
 *         compute_dense_block to compute the compressed matrix of
 *         a given Nystrom matrix that is fully described by these
 *         two routines.
 **/
template <typename Moments>
struct GalerkinMatrixEvaluatorDL {
  typedef typename Moments::eigenVector eigenVector;
  typedef typename Moments::eigenMatrix eigenMatrix;
  typedef typename eigenMatrix::Scalar value_type;
  const value_type cnst = 0.25 / FMCA_PI;
  GalerkinMatrixEvaluatorDL(const Moments &mom) : mom_(mom) {}
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
    double min_err = 10;
    double max_err = 0;
    for (auto j = 0; j < TC.indices().size(); ++j) {
      // set up first element
      const TriangularPanel &el1 = mom_.elements()[TC.indices()[j]];
      for (auto i = 0; i < TR.indices().size(); ++i) {
        // set up second element
        const TriangularPanel &el2 = mom_.elements()[TR.indices()[i]];
        // if two elements are not identical, we use a midpoint rule for
        // integration (we use an L2 normalization by the sqrt of the
        // volume element)
        if (TC.indices()[j] != TR.indices()[i]) {
          // el1 is the column element, el2 is the row element
          value_type nom = (el2.mp_ - el1.mp_).dot(el1.cs_.col(2));
          value_type r = std::pow((el2.mp_ - el1.mp_).norm(), 3.);
          (*retval)(i, j) =
              0.5 * sqrt(el1.volel_) * sqrt(el2.volel_) * cnst * nom / r;
          if (is_admissible(el1, el2)) {
#if 0
            const TriangularPanel el1 = TriangularPanel(
                Eigen::Vector3d({0, 0, 0}), Eigen::Vector3d({1, 0, 0}),
                Eigen::Vector3d({0, 1, 0}));
            const TriangularPanel el2 = TriangularPanel(
                Eigen::Vector3d({10, 0, 0}), Eigen::Vector3d({11, 0, 0}),
                Eigen::Vector3d({10, 1, 0}));
#endif
            double val = 0;
            double val2 = 0;
            for (auto k = 0; k < Rq_.xi.cols(); ++k) {
              const Eigen::Vector3d qp2 =
                  el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq_.xi.col(k);
              val += Rq_.w(k) * analyticIntD(el1, qp2);
              for (auto l = 0; l < Rq_.xi.cols(); ++l) {
                const Eigen::Vector3d qp1 =
                    el1.affmap_.col(0) +
                    el1.affmap_.rightCols(2) * Rq_.xi.col(k);
                value_type nom = (qp2 - qp1).dot(el1.cs_.col(2));
                value_type r = std::pow((qp2 - qp1).norm(), 3.);
                val2 += Rq_.w(k) * Rq_.w(l) * sqrt(2 * el1.volel_) *
                        sqrt(2 * el2.volel_) * cnst * nom / r;
              }
            }
            val2 = (*retval)(i, j);
            val *= 2 * cnst * sqrt(el2.volel_) / sqrt(el1.volel_);
            min_err = min_err > abs(val2 - val) / abs(val)
                          ? abs(val2 - val) / abs(val)
                          : min_err;
            max_err = max_err < abs(val2 - val) / abs(val)
                          ? abs(val2 - val) / abs(val)
                          : max_err;
            if (abs(val2 - val) / abs(val) > 2) {
              std::cout << val << " " << val2 << " " << nom << " " << r
                        << "dst: " << (el1.mp_ - el2.mp_).norm() << std::endl;
            }
          }
        } else {
          // if the elements are identical, we use the semi-analytic rule
          // from Steinbach and Rjasanow
          (*retval)(i, j) = 0;
          for (auto k = 0; k < Rq_.xi.cols(); ++k) {
            const Eigen::Vector3d qp =
                el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq_.xi.col(k);
            (*retval)(i, j) += Rq_.w(k) * analyticIntD(el2, qp);
          }
          (*retval)(i, j) *= 2 * cnst * sqrt(el2.volel_) / sqrt(el1.volel_);
        }
      }
    }
    std::cout << "min_err: " << min_err << " max_err: " << max_err << std::endl;
    return;
  }
  const Moments &mom_;
  const Quad::Quadrature<Quad::Radon> Rq_;
};

}  // namespace FMCA
#endif

//////////////////////////////////////////////////////////////////////
#if 0
          double val = 0;
          double val2 = 0;
          for (auto k = 0; k < Rq_.xi.cols(); ++k) {
            const Eigen::Vector3d qp =
                el2.affmap_.col(0) + el2.affmap_.rightCols(2) * Rq_.xi.col(k);
            val += Rq_.w(k) * analyticIntS(el1, qp);
            for (auto l = 0; l < Rq_.xi.cols(); ++l) {
              const Eigen::Vector3d qp2 =
                  el1.affmap_.col(0) + el1.affmap_.rightCols(2) * Rq_.xi.col(k);
              val2 += Rq_.w(k) * Rq_.w(l) / (qp - qp2).norm();
            }
          }
          val2 *= 2 * cnst * sqrt(el1.volel_) * sqrt(el2.volel_);
          val *= 2 * cnst * sqrt(el2.volel_) / sqrt(el1.volel_);
          max_err = (max_err < abs(val2 - val) / val) ? abs(val2 - val) / val
                                                      : max_err;
#endif
   //////////////////////////////////////////////////////////////////////

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
#ifndef FMCA_BEM_GALERKINRHSEVALUATOR_H_
#define FMCA_BEM_GALERKINRHSEVALUATOR_H_

namespace FMCA {

template <typename Moments>
struct GalerkinRHSEvaluator {
  typedef typename Moments::eigenVector eigenVector;
  typedef typename Moments::eigenMatrix eigenMatrix;
  typedef typename eigenMatrix::Scalar value_type;

  GalerkinRHSEvaluator(const Moments &mom) : mom_(mom) {}

  template <typename Derived, typename Functor>
  void compute_rhs(const ClusterTreeBase<Derived> &TR, const Functor &fun) {
    rhs_.resize(TR.indices().size());
    rhs_.setZero();
    for (auto i = 0; i < TR.indices().size(); ++i) {
      // set up element
      const TriangularPanel &el = mom_.elements()[TR.indices()[i]];
      // perform quadrature
      for (auto k = 0; k < Rq_.xi.cols(); ++k) {
        const Eigen::Vector3d qp =
            el.affmap_.col(0) + el.affmap_.rightCols(2) * Rq_.xi.col(k);
        rhs_(i) += Rq_.w(k) * fun(qp);
      }
      rhs_(i) *= el.volel_;
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Moments &mom_;
  const Quad::Quadrature<Quad::Radon> Rq_;
  eigenVector rhs_;
};

}  // namespace FMCA
#endif

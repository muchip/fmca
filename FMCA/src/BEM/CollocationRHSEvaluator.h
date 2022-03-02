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
#ifndef FMCA_BEM_COLLOCATIONRHSEVALUATOR_H_
#define FMCA_BEM_COLLOCATIONRHSEVALUATOR_H_

namespace FMCA {

template <typename Moments>
struct CollocationRHSEvaluator {
  typedef typename Moments::eigenVector eigenVector;
  typedef typename Moments::eigenMatrix eigenMatrix;
  typedef typename eigenMatrix::Scalar value_type;

  CollocationRHSEvaluator(const Moments &mom) : mom_(mom) {}

  template <typename Derived, typename Functor>
  void compute_rhs(const ClusterTreeBase<Derived> &TR, const Functor &fun) {
    rhs_.resize(TR.indices().size());
    rhs_.setZero();
    for (auto i = 0; i < TR.indices().size(); ++i) {
      // set up element
      const TriangularPanel &el = mom_.elements()[TR.indices()[i]];
      rhs_(i) = 0.5 * fun(el.mp_) * sqrt(2 * el.volel_);
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Moments &mom_;
  eigenVector rhs_;
};

}  // namespace FMCA
#endif

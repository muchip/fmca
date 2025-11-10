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
#ifndef FMCA_BEM_SLCOLLOCATIONPOTENTIALEVALUATOR_H_
#define FMCA_BEM_SLCOLLOCATIONPOTENTIALEVALUATOR_H_

namespace FMCA {

template <typename Moments>
struct SLCollocationPotentialEvaluator {
  const double cnst = 0.25 / FMCA_PI;
  SLCollocationPotentialEvaluator(const Moments &mom) : mom_(mom) {}

  template <typename Derived1, typename Derived2, typename Derived3>
  Vector compute(const ClusterTreeBase<Derived1> &TR,
                 const MatrixBase<Derived2> &rho,
                 const MatrixBase<Derived3> &P) {
    Vector pot(P.cols());
    pot.setZero();
    for (auto i = 0; i < P.cols(); ++i)
      for (auto j = 0; j < TR.block_size(); ++j) {
        // set up element
        const TriangularPanel &el = mom_.elements()[TR.indices()[j]];
        double r = (P.col(i) - el.mp_).norm();
        pot(i) += 0.5 * rho(j) / r * cnst * sqrt(2 * el.volel_);
      }
    return pot;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Moments &mom_;
};

}  // namespace FMCA
#endif

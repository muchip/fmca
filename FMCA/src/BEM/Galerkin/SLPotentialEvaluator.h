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
#ifndef FMCA_BEM_SLPOTENTIALEVALUATOR_H_
#define FMCA_BEM_SLPOTENTIALEVALUATOR_H_

namespace FMCA {

template <typename Moments> struct SLPotentialEvaluator {
  typedef typename Moments::eigenVector eigenVector;
  typedef typename Moments::eigenMatrix eigenMatrix;
  const double cnst = 0.25 / FMCA_PI;
  SLPotentialEvaluator(const Moments &mom) : mom_(mom) {}

  template <typename Derived1, typename Derived2, typename Derived3>
  eigenVector compute(const ClusterTreeBase<Derived1> &TR,
                      const Eigen::MatrixBase<Derived2> &rho,
                      const Eigen::MatrixBase<Derived3> &P) {
    eigenVector pot(P.cols());
    pot.setZero();
    for (auto i = 0; i < P.cols(); ++i)
      for (auto j = 0; j < TR.indices().size(); ++j) {
        // set up element
        const TriangularPanel &el = mom_.elements()[TR.indices()[j]];
#if 0
        double r = (P.col(i) - el.mp_).norm();
        pot(i) += cnst * 0.5 * rho(j) / r / sqrt(0.5 * el.volel_) * el.volel_;

#endif
        // perform quadrature
        for (auto k = 0; k < Rq_.xi.cols(); ++k) {
          const Eigen::Vector3d qp =
              el.affmap_.col(0) + el.affmap_.rightCols(2) * Rq_.xi.col(k);
          double r = (P.col(i) - qp).norm();
          pot(i) += Rq_.w(k) * rho(j) / r * cnst * sqrt(2 * el.volel_);
        }
      }
    return pot;
  }
  //////////////////////////////////////////////////////////////////////////////
  const Moments &mom_;
  const Quad::Quadrature<Quad::Radon> Rq_;
};

} // namespace FMCA
#endif

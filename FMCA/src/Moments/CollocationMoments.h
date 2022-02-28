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
#ifndef FMCA_MOMENTS_MOMENTCOMPUTER_GALERKIN_H_
#define FMCA_MOMENTS_MOMENTCOMPUTER_GALERKIN_H_

namespace FMCA {
template <typename Interpolator>
class GalerkinMoments {
 public:
  typedef typename Interpolator::eigenVector eigenVector;
  typedef typename Interpolator::eigenMatrix eigenMatrix;
  GalerkinMoments(const eigenMatrix &V, const eigenMatrix &F,
                  IndexType polynomial_degree = 3)
      : V_(V), F_(F), polynomial_degree_(polynomial_degree) {
    interp_.init(V_.cols(), polynomial_degree_);
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  typename otherDerived::eigenMatrix moment_matrix(
      const ClusterTreeBase<otherDerived> &CT) const {
    const otherDerived &H2T = CT.derived();
    typename otherDerived::eigenMatrix retval(interp_.Xi().cols(),
                                              H2T.indices().size());
    for (auto i = 0; i < H2T.indices().size(); ++i)
      retval.col(i) = interp_.evalPolynomials(
          ((P_.col(H2T.indices()[i]) - H2T.bb().col(0)).array() /
           H2T.bb().col(2).array())
              .matrix());
    return retval;
  }

  IndexType polynomial_degree() const { return polynomial_degree_; }
  const Interpolator &interp() const { return interp_; }
  const eigenMatrix &P() const { return P_; }

 private:
  const eigenMatrix &V_;
  const eigenMatrix &F_;
  IndexType polynomial_degree_;
  Interpolator interp_;
};

}  // namespace FMCA
#endif

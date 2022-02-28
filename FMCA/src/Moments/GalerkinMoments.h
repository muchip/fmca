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
template <typename Interpolator> class GalerkinMoments {
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
  eigenMatrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    const otherDerived &H2T = CT.derived();
    eigenMatrix retval(interp_.Xi().cols(), H2T.indices().size());
    for (auto i = 0; i < H2T.indices().size(); ++i) {
      Eigen::Matrix<FloatType, 3, 3> element;
      element.col(0) = V_.row(F_(H2T.indices()[i], 0)).transpose();
      element.col(1) = V_.row(F_(H2T.indices()[i], 1)).transpose();
      element.col(2) = V_.row(F_(H2T.indices()[i], 2)).transpose();
      element.rightCols(2).colwise() -= element.col(0);
      FloatType el_area = 0.5 * (element.col(1)).cross(element.col(2)).norm();
      for (auto j = 0; j < Rq_.xi.cols(); ++j) {
        // map quadrature point to element
        Eigen::Matrix<FloatType, 3, 1> qp =
            element.rightCols(2) * Rq_.xi.col(j) + element.col(0);
        retval.col(i) =
            Rq_.w(j) * sqrt(el_area) *
            interp_.evalPolynomials(
                ((qp - H2T.bb().col(0)).array() / H2T.bb().col(2).array())
                    .matrix());
      }
    }
    return retval;
  }

  IndexType polynomial_degree() const { return polynomial_degree_; }
  const Interpolator &interp() const { return interp_; }
  const eigenMatrix &V() const { return V_; }
  const eigenMatrix &F() const { return F_; }

private:
  const eigenMatrix &V_;
  const eigenMatrix &F_;
  const Quad::Quadrature<Quad::Radon> Rq_;
  IndexType polynomial_degree_;
  Interpolator interp_;
};

} // namespace FMCA
#endif

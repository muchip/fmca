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
#ifndef FMCA_MOMENTS_SAMPLETMOMENTCOMPUTER_GALERKIN_H_
#define FMCA_MOMENTS_SAMPLETMOMENTCOMPUTER_GALERKIN_H_

namespace FMCA {

/**
 *  \brief implements the moments for the Galerkin method
 *
 **/
template <typename Interpolator>
class GalerkinSampletMoments : public GalerkinMoments<Interpolator> {
public:
  using Base = GalerkinMoments<Interpolator>;
  using Base::elements;
  using Base::interp;
  GalerkinSampletMoments(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                         Index polynomial_degree = 3)
      : Base(V, F, SampletHelper::internal_q(polynomial_degree, V.cols())),
        polynomial_degree_(polynomial_degree) {
    multinomial_coeffs_ = SampletHelper::multinomialCoefficientMatrix<
        Matrix, MultiIndexSet<TotalDegree>>(Base::interp().idcs());
    mq_ = binomialCoefficient(V.cols() + polynomial_degree_, V.cols());
    mq2_ = binomialCoefficient(V.cols() + Base::polynomial_degree(), V.cols());
  }
  //////////////////////////////////////////////////////////////////////////////
  Index polynomial_degree() const { return polynomial_degree_; }
  Index mdtilde() const { return mq_; }
  Index mdtilde2() const { return mq2_; }
  const Matrix &multinomial_coeffs() const { return multinomial_coeffs_; }
  Index internal_polynomial_degree() const { return Base::polynomial_degree(); }
  Matrix shift_matrix(const Matrix &shift) const {
    return SampletHelper::monomialMomentShifter(shift, Base::interp().idcs(),
                                                multinomial_coeffs_);
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  Matrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    Matrix mp = 0.5 * (CT.bb().col(0) + CT.bb().col(1));
    Matrix retval(interp().Xi().cols(), CT.indices().size());
    for (auto i = 0; i < CT.indices().size(); ++i) {
      const TriangularPanel &el = elements()[CT.indices()[i]];
      retval.col(i).setZero();
      for (auto j = 0; j < Rq_.xi.cols(); ++j) {
        // map quadrature point to element
        const Eigen::Vector3d qp =
            el.affmap_.col(0) + el.affmap_.rightCols(2) * Rq_.xi.col(j);
        retval.col(i) += Rq_.w(j) * interp().evalPolynomials(qp - mp);
      }
      retval.col(i) *= sqrt(2 * el.volel_);
    }
    return retval;
  }

private:
  const Quad::Quadrature<Quad::Radon> Rq_;
  Index polynomial_degree_;
  Index mq_;
  Index mq2_;
  Matrix multinomial_coeffs_;
};

} // namespace FMCA
#endif

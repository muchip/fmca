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
#ifndef FMCA_MOMENTS_MOMENTCOMPUTER_NYSTROMSAMPLET_H_
#define FMCA_MOMENTS_MOMENTCOMPUTER_NYSTROMSAMPLET_H_

namespace FMCA {
template <typename Interpolator>
class NystromSampletMoments : public NystromMoments<Interpolator> {
public:
  using Base = NystromMoments<Interpolator>;
  using Base::interp;
  using Base::P;

  NystromSampletMoments(const Matrix &P, Index polynomial_degree = 2)
      : Base(P, SampletHelper::internal_q(polynomial_degree, P.rows())),
        polynomial_degree_(polynomial_degree) {
    multinomial_coeffs_ = SampletHelper::multinomialCoefficientMatrix<
        Matrix, MultiIndexSet<TotalDegree>>(Base::interp().idcs());
    mq_ = binomialCoefficient(P.rows() + polynomial_degree_, P.rows());
    mq2_ = binomialCoefficient(P.rows() + Base::polynomial_degree(), P.rows());
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

  template <typename otherDerived>
  Matrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    Matrix mp = 0.5 * (CT.bb().col(0) + CT.bb().col(1));
    Matrix retval(interp().Xi().cols(), CT.indices().size());
    for (auto i = 0; i < CT.indices().size(); ++i)
      retval.col(i) = interp().evalPolynomials(P().col(CT.indices()[i]) - mp);
    return retval;
  }

private:
  Index polynomial_degree_;
  Index mq_;
  Index mq2_;
  Matrix multinomial_coeffs_;
};
} // namespace FMCA
#endif

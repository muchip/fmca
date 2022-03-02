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
#ifndef FMCA_MOMENTS_SAMPLETMOMENTCOMPUTER_COLLOCATION_H_
#define FMCA_MOMENTS_SAMPLETMOMENTCOMPUTER_COLLOCATION_H_

namespace FMCA {

/**
 *  \brief implements the moments for the Galerkin method
 *
 **/
template <typename Interpolator>
class CollocationSampletMoments : public CollocationMoments<Interpolator> {
 public:
  using Base = CollocationMoments<Interpolator>;
  using Base::elements;
  using Base::interp;
  typedef typename Base::eigenVector eigenVector;
  typedef typename Base::eigenMatrix eigenMatrix;
  CollocationSampletMoments(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                            IndexType polynomial_degree = 3)
      : Base(V, F, SampletHelper::internal_q(polynomial_degree, V.cols())),
        polynomial_degree_(polynomial_degree) {
    multinomial_coeffs_ = SampletHelper::multinomialCoefficientMatrix<
        eigenMatrix, MultiIndexSet<TotalDegree>>(Base::interp().idcs());
    mq_ = binomialCoefficient(V.cols() + polynomial_degree_, V.cols());
    mq2_ = binomialCoefficient(V.cols() + Base::polynomial_degree(), V.cols());
  }
  //////////////////////////////////////////////////////////////////////////////
  IndexType polynomial_degree() const { return polynomial_degree_; }
  IndexType mdtilde() const { return mq_; }
  IndexType mdtilde2() const { return mq2_; }
  const eigenMatrix &multinomial_coeffs() const { return multinomial_coeffs_; }
  IndexType internal_polynomial_degree() const {
    return Base::polynomial_degree();
  }
  eigenMatrix shift_matrix(const eigenMatrix &shift) const {
    return SampletHelper::monomialMomentShifter(shift, Base::interp().idcs(),
                                                multinomial_coeffs_);
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  eigenMatrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    eigenMatrix mp = 0.5 * (CT.bb().col(0) + CT.bb().col(1));
    eigenMatrix retval(interp().Xi().cols(), CT.indices().size());
    for (auto i = 0; i < CT.indices().size(); ++i) {
      const TriangularPanel &el = elements()[CT.indices()[i]];
      retval.col(i) =
          0.5 * interp().evalPolynomials(el.mp_ - mp) * sqrt(2 * el.volel_);
    }
    return retval;
  }

 private:
  IndexType polynomial_degree_;
  IndexType mq_;
  IndexType mq2_;
  eigenMatrix multinomial_coeffs_;
};

}  // namespace FMCA
#endif

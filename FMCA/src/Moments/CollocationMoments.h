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
#ifndef FMCA_MOMENTS_MOMENTCOMPUTER_COLLOCATION_H_
#define FMCA_MOMENTS_MOMENTCOMPUTER_COLLOCATION_H_

namespace FMCA {

/**
 *  \brief implements the moments for the Galerkin method
 *
 **/
template <typename Interpolator>
class CollocationMoments {
 public:
  typedef typename Interpolator::eigenVector eigenVector;
  typedef typename Interpolator::eigenMatrix eigenMatrix;
  CollocationMoments(const Matrix &V, const iMatrix &F,
                     Index polynomial_degree = 3)
      : V_(V), F_(F), polynomial_degree_(polynomial_degree) {
    interp_.init(V_.cols(), polynomial_degree_);
    elements_.clear();
    for (auto i = 0; i < F_.rows(); ++i)
      elements_.emplace_back(TriangularPanel(V_.row(F_(i, 0)).transpose(),
                                             V_.row(F_(i, 1)).transpose(),
                                             V_.row(F_(i, 2)).transpose()));
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  eigenMatrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    const otherDerived &H2T = CT.derived();
    eigenMatrix retval(interp_.Xi().cols(), H2T.indices().size());
    for (auto i = 0; i < H2T.indices().size(); ++i) {
      const TriangularPanel &el = elements_[H2T.indices()[i]];
      retval.col(i) =
          0.5 *
          interp_.evalPolynomials(
              ((el.mp_ - H2T.bb().col(0)).array() / H2T.bb().col(2).array())
                  .matrix()) *
          sqrt(2 * el.volel_);
    }
    return retval;
  }

  Index polynomial_degree() const { return polynomial_degree_; }
  const Interpolator &interp() const { return interp_; }
  const Matrix &V() const { return V_; }
  const iMatrix &F() const { return F_; }
  const std::vector<TriangularPanel> &elements() const { return elements_; }

 private:
  const Matrix &V_;
  const iMatrix &F_;
  std::vector<TriangularPanel> elements_;
  Index polynomial_degree_;
  Interpolator interp_;
};

}  // namespace FMCA
#endif

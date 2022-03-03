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

/**
 *  \brief implements the moments for the Galerkin method
 *
 **/
template <typename Interpolator>
class GalerkinMoments {
 public:
  typedef typename Interpolator::eigenVector eigenVector;
  typedef typename Interpolator::eigenMatrix eigenMatrix;
  GalerkinMoments(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                  IndexType polynomial_degree = 3)
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
      retval.col(i).setZero();
      for (auto j = 0; j < Rq_.xi.cols(); ++j) {
        // map quadrature point to element
        const Eigen::Vector3d qp =
            el.affmap_.col(0) + el.affmap_.rightCols(2) * Rq_.xi.col(j);
        retval.col(i) +=
            Rq_.w(j) * interp_.evalPolynomials(((qp - H2T.bb().col(0)).array() /
                                                H2T.bb().col(2).array())
                                                   .matrix());
      }
      retval.col(i) *= sqrt(2 * el.volel_);
    }
    return retval;
  }

  IndexType polynomial_degree() const { return polynomial_degree_; }
  const Interpolator &interp() const { return interp_; }
  const Eigen::MatrixXd &V() const { return V_; }
  const Eigen::MatrixXi &F() const { return F_; }
  const std::vector<TriangularPanel> &elements() const { return elements_; }

 private:
  const Eigen::MatrixXd &V_;
  const Eigen::MatrixXi &F_;
  const Quad::Quadrature<Quad::Radon> Rq_;
  std::vector<TriangularPanel> elements_;
  IndexType polynomial_degree_;
  Interpolator interp_;
};

}  // namespace FMCA
#endif

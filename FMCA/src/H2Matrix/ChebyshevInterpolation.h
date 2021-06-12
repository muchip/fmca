#ifndef FMCA_H2MATRIX_CHEBYSHEVINTERPOLATION_H_
#define FMCA_H2MATRIX_CHEBYSHEVINTERPOLATION_H_

namespace FMCA {
/**
 *  \ingroup H2Matrix
 *  \brief Provide functionality for Barycentric Lagrange interpolation
 *         at Chebyshev nodes
 *
 *  The formulas stem from
 *  [1] Berrut, Trefethen: Barycentric Lagrange Interpolation,
 *      SIAM Review 46(3):501-517, 2004*
 *  [2] Higham: The numerical stability of barycentric Lagrange interpolation,
 *      IMA Journal of Numerical Analysis 24:547-556, 2004
 */

/**
 *  \brief These are the classical Chebyshev nodes rescaled to [0,1]
 **/
template <typename ValueType, IndexType Dim, IndexType Deg>
class TensorProductInterpolator {
public:
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, 1> eigenVector;
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  /**
   *  \brief These are the corresponding weights of the Chebyshev nodes
   *         for barycentric interpolation. see [1]. Note: The scaling is wrong
   *         as the nodes are on [0,1]. However, this does not matter as
   *         the factor cancels.
   **/
  void init() {
    const ValueType weight = 2. * FMCA_PI / (2. * ValueType(Deg) + 2.);
    // univariate Chebyshev nodes
    xi_ = (weight * (eigenVector::LinSpaced(Deg + 1, 0, Deg).array() + 0.5))
              .cos();
    xi_.array() = 0.5 * (xi_.array() + 1);
    // univariate barycentric weights
    w_ = (weight * (eigenVector::LinSpaced(Deg + 1, 0, Deg).array() + 0.5))
             .sin();
    for (auto i = 1; i < w_.size(); i += 2)
      w_(i) *= -1.;
    idcs_.init(Deg);
    TP_xi_.resize(Dim, idcs_.get_MultiIndexSet().size());
    // determine tensor product interpolation points
    IndexType k = 0;
    for (const auto &it : idcs_.get_MultiIndexSet()) {
      for (auto i = 0; i < it.size(); ++i) {
        TP_xi_(i, k) = xi_(it[i]);
      }
      ++k;
    }
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  eigenVector evalLagrangePolynomials(const Eigen::MatrixBase<Derived> &pt) {
    eigenVector retval(idcs_.get_MultiIndexSet().size());
    eigenVector weight(Dim);
    eigenVector my_pt = pt.col(0);
    retval.setOnes();
    IndexType inf_counter = 0;
    for (auto i = 0; i < Dim; ++i)
      weight(i) = (w_.array() / (my_pt(i) - xi_.array())).sum();
    IndexType k = 0;
    for (const auto &it : idcs_.get_MultiIndexSet()) {
      for (auto i = 0; i < Dim; ++i)
        if (abs(my_pt(i) - xi_(it[i])) > FMCA_ZERO_TOLERANCE)
          retval(k) *= w_(it[i]) / (my_pt(i) - xi_(it[i])) / weight(i);
      ++k;
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &get_Xi() const { return TP_xi_; }

private:
  MultiIndexSet<Dim, TensorProduct> idcs_;
  eigenMatrix TP_xi_;
  eigenVector xi_;
  eigenVector w_;
};
} // namespace FMCA
#endif

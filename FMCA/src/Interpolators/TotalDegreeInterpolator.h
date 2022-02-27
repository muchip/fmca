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
#ifndef FMCA_H2MATRIX_TOTALDEGREEINTERPOLATOR_H_
#define FMCA_H2MATRIX_TOTALDEGREEINTERPOLATOR_H_

namespace FMCA {

/**
 *  \brief These are Leja points on [0,1] computed by solving the optimization
 *         problem brute force.
 **/
const double LejaPoints[50] = {
    0.000000000000000, 1.000000000000000, 0.500000000000000, 0.788678451268611,
    0.170635616774738, 0.919632647436922, 0.064990192286888, 0.347189510436603,
    0.660868726719188, 0.971486942715307, 0.023646888739014, 0.260272622930690,
    0.856320986416747, 0.577974917178507, 0.112564077843152, 0.989746467119162,
    0.419414129035433, 0.008323389969770, 0.730685392807863, 0.945948035279350,
    0.214038884354495, 0.043710396865931, 0.824627754934647, 0.460089106589785,
    0.621157280776602, 0.138850053366516, 0.996342849337513, 0.304601946962690,
    0.891065764497292, 0.002956807141683, 0.698791456288239, 0.086803561718841,
    0.960022570238344, 0.382408454309738, 0.540598624130312, 0.032588364089015,
    0.982100299053592, 0.762224445021450, 0.235884153848761, 0.874719860869950,
    0.014592513360260, 0.190047332131628, 0.932663600770070, 0.599440054937718,
    0.325638738643865, 0.998719486196507, 0.075661046964471, 0.806873625853102,
    0.440350654950455, 0.001046782061681};

template <typename ValueType> class TotalDegreeInterpolator {
public:
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, 1> eigenVector;
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  /**
   *  \brief These are the corresponding weights of the Chebyshev nodes
   *         for barycentric interpolation. see [1]. Note: The scaling is wrong
   *         as the nodes are on [0,1]. However, this does not matter as
   *         the factor cancels.
   **/
  void init(IndexType dim, IndexType deg) {
    dim_ = dim;
    deg_ = deg;
    idcs_.init(dim, deg);
    TD_xi_.resize(dim_, idcs_.get_MultiIndexSet().size());
    V_.resize(idcs_.get_MultiIndexSet().size(),
              idcs_.get_MultiIndexSet().size());
    // determine tensor product interpolation points
    IndexType k = 0;
    for (const auto &it : idcs_.get_MultiIndexSet()) {
      for (auto i = 0; i < it.size(); ++i)
        TD_xi_(i, k) = LejaPoints[it[i]];
      V_.row(k) = evalPolynomials(TD_xi_.col(k)).transpose();
      ++k;
    }
    invV_ = V_.inverse();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  eigenMatrix
  evalLegendrePolynomials1D(const Eigen::MatrixBase<Derived> &pt) const {
    eigenMatrix retval(pt.rows(), deg_ + 1);
    eigenVector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setZero();
    P1.setOnes();
    retval.col(0) = P1;
    for (auto i = 1; i <= deg_; ++i) {
      retval.col(i) = ValueType(2 * i - 1) / ValueType(i) *
                          (2 * pt.array() - 1) * P1.array() -
                      ValueType(i - 1) / ValueType(i) * P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename Derived>
  eigenMatrix evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    eigenVector retval(idcs_.get_MultiIndexSet().size());
    eigenMatrix p_values = evalLegendrePolynomials1D(pt);
    retval.setOnes();
    IndexType k = 0;
    for (const auto &it : idcs_.get_MultiIndexSet()) {
      for (auto i = 0; i < dim_; ++i)
        retval(k) *= p_values(i, it[i]);
      ++k;
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &Xi() const { return TD_xi_; }
  const eigenMatrix &invV() const { return invV_; }
  const eigenMatrix &V() const { return V_; }

private:
  MultiIndexSet<TotalDegree> idcs_;
  eigenMatrix TD_xi_;
  eigenMatrix invV_;
  eigenMatrix V_;
  IndexType dim_;
  IndexType deg_;
};
} // namespace FMCA
#endif

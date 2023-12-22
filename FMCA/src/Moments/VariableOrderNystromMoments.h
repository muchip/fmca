// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_MOMENTS_VARIABLEORDERMOMENTCOMPUTER_NYSTROM_H_
#define FMCA_MOMENTS_VARIABLEORDERMOMENTCOMPUTER_NYSTROM_H_

namespace FMCA {
template <typename Interpolator>
class VariableOrderNystromMoments {
 public:
  VariableOrderNystromMoments(const Matrix &P,
                              const iVector &polynomial_degrees)
      : P_(P), polynomial_degrees_(polynomial_degrees) {
    interp_.resize(polynomial_degrees_.size());
    for (Index i = 0; i < polynomial_degrees_.size(); ++i)
      interp_[i].init(P_.rows(), polynomial_degrees_(i));
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  Matrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    const otherDerived &H2T = CT.derived();
    const Index level = H2T.level();
    Matrix retval(interp_[level].Xi().cols(), H2T.block_size());
    for (auto i = 0; i < H2T.block_size(); ++i)
      retval.col(i) = interp_[level].evalPolynomials(
          ((P_.col(H2T.indices()[i]) - H2T.bb().col(0)).array() /
           H2T.bb().col(2).array())
              .matrix());
    return retval;
  }

  Index polynomial_degree(const Index level = 0) const {
    return polynomial_degrees_[level];
  }
  const Interpolator &interp(const Index level = 0) const {
    return interp_[level];
  }
  const Matrix &P() const { return P_; }

 private:
  const Matrix &P_;
  iVector polynomial_degrees_;
  std::vector<Interpolator> interp_;
};

}  // namespace FMCA
#endif

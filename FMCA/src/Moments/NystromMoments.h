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
#ifndef FMCA_MOMENTS_MOMENTCOMPUTER_NYSTROM_H_
#define FMCA_MOMENTS_MOMENTCOMPUTER_NYSTROM_H_

namespace FMCA {
template <typename Interpolator>
class NystromMoments {
 public:
  NystromMoments(const Matrix &P, Index polynomial_degree = 3)
      : P_(P), polynomial_degree_(polynomial_degree) {
    interp_.init(P_.rows(), polynomial_degree_);
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  Matrix moment_matrix(const ClusterTreeBase<otherDerived> &CT) const {
    const otherDerived &H2T = CT.derived();
    Matrix retval(interp_.Xi().cols(), H2T.block_size());
    for (auto i = 0; i < H2T.block_size(); ++i)
      retval.col(i) = interp_.evalPolynomials(
          ((P_.col(H2T.indices()[i]) - H2T.bb().col(0)).array() /
           H2T.bb().col(2).array())
              .matrix());
    return retval;
  }

  Index polynomial_degree() const { return polynomial_degree_; }
  const Interpolator &interp() const { return interp_; }
  const Matrix &P() const { return P_; }

 private:
  const Matrix &P_;
  Index polynomial_degree_;
  Interpolator interp_;
};

}  // namespace FMCA
#endif

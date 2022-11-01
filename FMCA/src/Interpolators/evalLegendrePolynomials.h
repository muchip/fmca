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
namespace FMCA {
namespace internal {
//////////////////////////////////////////////////////////////////////////////
template <typename Derived>
Matrix evalLegendrePolynomials(Index deg,
                               const Eigen::MatrixBase<Derived> &pt) {
  Matrix retval(pt.rows(), deg + 1);
  Vector P0, P1;
  P0.resize(pt.rows());
  P1.resize(pt.rows());
  P0.setZero();
  P1.setOnes();
  retval.col(0) = P1;
  for (auto i = 1; i <= deg; ++i) {
    retval.col(i) =
        Scalar(2 * i - 1) / Scalar(i) * (2 * pt.array() - 1) * P1.array() -
        Scalar(i - 1) / Scalar(i) * P0.array();
    P0 = P1;
    P1 = retval.col(i);
    // L2-normalize
    retval.col(i) *= sqrt(2 * i + 1);
  }
  return retval;
}
}  // namespace internal
}  // namespace FMCA

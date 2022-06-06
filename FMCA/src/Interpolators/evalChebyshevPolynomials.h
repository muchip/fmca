// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
namespace FMCA {
//////////////////////////////////////////////////////////////////////////////
template <typename Derived>
Matrix evalChebyshevPolynomials(Index deg,
                                  const Eigen::MatrixBase<Derived> &pt) {
  Matrix retval(pt.rows(), deg + 1);
  retval.col(0).setOnes();
  if (deg < 1)
    return retval;
  retval.col(1).array() = 2 * pt.array() - 1;
  for (auto i = 2; i <= deg; ++i)
    retval.col(i).array() =
        2 * (2 * pt.array() - 1) * retval.col(i - 1).array() -
        retval.col(i - 2).array();
  return retval;
}

} // namespace FMCA

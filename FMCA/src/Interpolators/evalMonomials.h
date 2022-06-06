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
Matrix evalMonomials(Index deg, const Eigen::MatrixBase<Derived> &pt) {
  Matrix retval(pt.rows(), deg + 1);
  retval.col(0).setOnes();
  for (auto i = 1; i <= deg; ++i) {
    retval.col(i) = pt.array().pow(Scalar(i));
  }
  return retval;
}
} // namespace FMCA

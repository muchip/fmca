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
#ifndef FMCA_INTERPOLATORS_INTERPOLATIONPOINTSGENERATOR_H_
#define FMCA_INTERPOLATORS_INTERPOLATIONPOINTSGENERATOR_H_

#include <Eigen/QR>
#include <Eigen/SVD>

namespace FMCA {

template <typename Interpolator>
Matrix interpolationPointsGenerator(const Interpolator &interp) {
  Index dim = interp.dim();
  Index deg = interp.deg();
  Index n_pols = interp.idcs().index_set().size();
  Index n_cand_pts = 2 * n_pols;
  Matrix halton_pts(dim, n_cand_pts);
  Matrix V(n_cand_pts, n_pols);
  Matrix retval;
  HaltonSet<100> hs(dim);
  std::cout << n_pols << " <- n_pols" << std::endl;
  for (auto i = 0; i < halton_pts.cols(); ++i) {
    halton_pts.col(i) = hs.EigenHaltonVector();
    hs.next();
  }
  for (auto i = 0; i < n_cand_pts; ++i)
    V.row(i) = interp.evalPolynomials(halton_pts.col(i)).transpose();
  {
    Eigen::HouseholderQR<Matrix> qr;
    qr.compute(V);
    V = qr.householderQ() * Matrix::Identity(n_cand_pts, n_pols);
  }
  Eigen::ColPivHouseholderQR<Matrix> qr;
  qr.compute(V.transpose());
  const auto &idcs = qr.colsPermutation().indices();
  retval.resize(dim, n_pols);
  for (auto i = 0; i < n_pols; ++i)
    retval.col(i) = halton_pts.col(idcs(i));
  V.resize(n_pols, n_pols);
  for (auto i = 0; i < n_pols; ++i)
    V.row(i) = interp.evalPolynomials(retval.col(i)).transpose();
  Eigen::JacobiSVD<Matrix> svd;
  svd.compute(V);
  std::cout << "condition: "
            << svd.singularValues().maxCoeff() / svd.singularValues().minCoeff()
            << std::endl;
  return retval;
}
} // namespace FMCA
#endif

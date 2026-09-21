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
#ifndef FMCA_INTERPOLATOR_DEMARCHIPOINTS_H_
#define FMCA_INTERPOLATOR_DEMARCHIPOINTS_H_

#include "../util/HaltonSet.h"
#include "evalPolynomials.h"

namespace FMCA {
/**
 *  \brief These are DeMarchi points on [0,1]^d
 **/
template <typename MultiIndexSet>
Matrix DeMarchiPoints(const MultiIndexSet &idcs,
                      const Index oversmplng_factor = 10) {
  const unsigned int dim = idcs.dim();
  const unsigned int n = idcs.index_set().size();
  Matrix retval(dim, n);
  Matrix VT(n, oversmplng_factor * n);
  // create matrix of Halton points
  Matrix Halton_pts(dim, oversmplng_factor * n);
  HaltonSet<100> hs(dim);
  for (Index i = 0; i < Halton_pts.cols(); ++i) {
    Halton_pts.col(i) = hs.MapHaltonVector();
    hs.next();
  }
  for (Index i = 0; i < Halton_pts.cols(); ++i)
    VT.col(i) = internal::evalPolynomials(idcs, Halton_pts.col(i));
  ColPivHouseholderQR qr;
  qr.compute(VT);
  const auto &pt_idcs = qr.colsPermutation().indices();
  for (Index i = 0; i < n; ++i) retval.col(i) = Halton_pts.col(pt_idcs(i));
  // compute condition of interpolation
  VT.resize(n, n);
  for (Index i = 0; i < retval.cols(); ++i)
    VT.col(i) = internal::evalPolynomials(idcs, retval.col(i));
  JacobiSVD svd;
  svd.compute(VT);
  if (svd.singularValues().maxCoeff() / svd.singularValues().minCoeff() > 1e6)
    std::cout << "Ill-conditioned DeMarchi points ("
              << svd.singularValues().maxCoeff() /
                     svd.singularValues().minCoeff()
              << ").\nConsider a different oversampling factor!" << std::endl;
  return retval;
}

}  // namespace FMCA

#endif

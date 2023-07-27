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
#ifndef FMCA_INTERPOLATORS_DEMARCHIPOINTS_H_
#define FMCA_INTERPOLATORS_DEMARCHIPOINTS_H_

#include "../util/HaltonSet.h"

namespace FMCA {
/**
 *  \brief These are DeMarchi points on [0,1]^d
 **/
template <typename MultiIndexSet>
Matrix DeMarchiPoints(const MultiIndexSet &idcs, Index oversmplng_factor = 10) {
  const unsigned int dim = idcs.dim();
  const unsigned int n = idcs.index_set().size();
  Matrix retval(dim, n);
  Matrix VT(n, oversampl_factor * n);
  // create matrix of Halton points
  Eigen::MatrixXd Halton_pts(dim, oversampl_factor * n);
  HaltonSet<100> hs(dim);
  for (Index i = 0; i < Halton_pts.cols(); ++i) {
    Halton_pts.col(i) = hs.EigenHaltonVector();
    hs.next();
  }
  for (Index i = 0; i < Halton_pts.cols(); ++i)
    retval.col(i) = evalPolynomials(pts.col(i), idcs).transpose();

  return retval;
}

}  // namespace FMCA

#endif

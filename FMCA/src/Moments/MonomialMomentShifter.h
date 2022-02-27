// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_MOMENTS_MONOMIALMOMENTSHIFTER_H_
#define FMCA_MOMENTS_MONOMIALMOMENTSHIFTER_H_

namespace FMCA {
/**
 *  \ingroup Moments
 *  \brief computes the transformation matrix from the son cluster moments
 *         to the dad cluster moments. This one is only based on polynomial
 *         interpolation and hence agnostic to the particular moment generation
 *         procedure
 **/
template <typename Matrix, typename MultiIndexSet>
Matrix monomialMomentShifter(Matrix &shift, const MultiIndexSet &idcs,
                             Matrix &mult_coeffs) {
  Matrix retval = mult_coeffs;
  if (shift.norm() < FMCA_ZERO_TOLERANCE)
    return Matrix::Identity(retval.rows(), retval.cols());
  IndexType i = 0;
  IndexType j = 0;
  typename Matrix::Scalar weight;
  for (const auto &it1 : idcs.get_MultiIndexSet()) {
    j = 0;
    for (const auto &it2 : idcs.get_MultiIndexSet()) {
      // check if the multinomial coefficient is non-zero
      if (retval(j, i)) {
        for (auto k = 0; k < shift.size(); ++k)
          // make sure that 0^0 = 1
          if (it2[k] - it1[k])
            retval(j, i) *= std::pow(shift(k), it2[k] - it1[k]);
      }
      ++j;
    }
    ++i;
  }
  return retval;
}
} // namespace FMCA
#endif

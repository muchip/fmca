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
#ifndef FMCA_MOMENTS_SAMPLETHELPER_H_
#define FMCA_MOMENTS_SAMPLETHELPER_H_

namespace FMCA {

namespace SampletHelper {

/**
 *  \ingroup Moments
 *  \brief computes the internal polynomial degree for the vanishing moments
 **/
inline IndexType internal_q(IndexType q, IndexType dim) {
  IndexType retval = q;
  IndexType mq = binomialCoefficient(dim + q, dim);
  IndexType mq2 = mq;

  while (2 * mq > mq2) {
    ++retval;
    mq2 = binomialCoefficient(dim + retval, dim);
  }
  return retval;
}

/**
 *  \ingroup Moments
 *  \brief computes the transformation matrix from the son cluster moments
 *         to the dad cluster moments. This one is only based on polynomial
 *         interpolation and hence agnostic to the particular moment generation
 *         procedure
 **/
template <typename Derived, typename MultiIndexSet, typename Derived2>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
monomialMomentShifter(const Eigen::MatrixBase<Derived> &shift,
                      const MultiIndexSet &idcs,
                      const Eigen::MatrixBase<Derived2> &mult_coeffs) {
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
      retval = mult_coeffs;
  if (shift.norm() < FMCA_ZERO_TOLERANCE)
    return Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic,
                         Eigen::Dynamic>::Identity(retval.rows(),
                                                   retval.cols());
  IndexType i = 0;
  IndexType j = 0;
  typename Derived::Scalar weight;
  for (const auto &it1 : idcs.index_set()) {
    j = 0;
    for (const auto &it2 : idcs.index_set()) {
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

/**
 *  \ingroup Moments
 *  \brief computes a matrix containing all possible multinomial
 *         combinations for a given multi index set
 **/
template <typename Matrix, typename MultiIndexSet>
inline Matrix multinomialCoefficientMatrix(const MultiIndexSet &idcs) {
  Matrix retval(idcs.index_set().size(), idcs.index_set().size());
  IndexType i = 0;
  IndexType j = 0;
  for (const auto &beta : idcs.index_set()) {
    for (const auto &alpha : idcs.index_set()) {
      retval(j, i) = multinomialCoefficient(alpha, beta);
      ++j;
    }
    ++i;
    j = 0;
  }
  return retval;
}
} // namespace SampletHelper
} // namespace FMCA
#endif

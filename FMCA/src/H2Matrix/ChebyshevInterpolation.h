#ifndef FMCA_H2MATRIX_CHEBYSHEVINTERPOLATION_H_
#define FMCA_H2MATRIX_CHEBYSHEVINTERPOLATION_H_

namespace FMCA {
/**
 *  \ingroup H2Matrix
 *  \brief Provide functionality for Barycentric Lagrange interpolation
 *         at Chebyshev nodes
 *
 *  The formulas stem from
 *  [1] Berrut, Trefethen: Barycentric Lagrange Interpolation,
 *      SIAM Review 46(3):501-517, 2004*
 *  [2] Higham: The numerical stability of barycentric Lagrange interpolation,
 *      IMA Journal of Numerical Analysis 24:547-556, 2004
 */

/**
 *  \brief These are the classical Chebyshev nodes rescaled to [0,1]
 **/
template <typename ValueType>
Eigen::Matrix<ValueType, Eigen::Dynamic, 1> ChebyshevNodes(IndexType n) {
  const ValueType weight = 2. * FMCA_PI / (2. * ValueType(n) + 2.);
  return 0.5 *
         ((weight *
           (Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::LinSpaced(n + 1, 0, n)
                .array() +
            0.5))
              .cos() +
          1.);
}

/**
 *  \brief These are the corresponding weights of the Chebyshev nodes
 *         for barycentric interpolation. see [1]. Note: The scaling is wrong
 *         as the nodes are on [0,1]. However, this does not matter as
 *         the factor cancels.
 **/
template <typename ValueType>
Eigen::Matrix<ValueType, Eigen::Dynamic, 1> ChebyshevWeights(IndexType n) {
  const ValueType weight = 2. * FMCA_PI / (2. * ValueType(n) + 2.);
  Eigen::Matrix<ValueType, Eigen::Dynamic, 1> retval =
      (weight *
       (Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::LinSpaced(n + 1, 0, n)
            .array() +
        0.5))
          .sin();
  // next one is for legacy Eigen... newer version should have seq indexing
  for (auto i = 1; i < retval.size(); i += 2)
    retval(i) *= -1.;
  return retval;
}
} // namespace FMCA
#endif

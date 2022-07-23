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
#ifndef FMCA_SAMPLETS_MOMENTCOMPUTER_H_
#define FMCA_SAMPLETS_MOMENTCOMPUTER_H_

namespace FMCA {

namespace internal {
/**
 *  \ingroup internal
 *  \brief computes the transformation matrix from the son cluster moments
 *         to the dad cluster moments. This one is only based on polynomial
 *         interpolation and hence agnostic to the particular moment generation
 *         procedure
 **/
template <typename Derived, typename MultiIndexSet>
typename traits<Derived>::eigenMatrix momentShifter(
    const typename traits<Derived>::eigenMatrix &shift,
    const MultiIndexSet &idcs,
    const typename traits<Derived>::eigenMatrix &mult_coeffs) {
  typedef typename traits<Derived>::eigenMatrix eigenMatrix;
  eigenMatrix retval = mult_coeffs;
  if (shift.norm() < FMCA_ZERO_TOLERANCE)
    return eigenMatrix::Identity(retval.rows(), retval.cols());
  IndexType i = 0;
  IndexType j = 0;
  typename traits<Derived>::value_type weight;
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
}  // namespace internal

/**
 *  \brief provides the methods for the moment computation in case
 *         of moments for sample data
 **/
template <typename Derived, typename MultiIndexSet>
class SampleMomentComputer {
 public:
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  void init(IndexType dim, IndexType dtilde) {
    // set desired number of vanishing moments
    dim_ = dim;
    dtilde_ = dtilde;
    m_dtilde_ = binomialCoefficient(dim_ + dtilde_ - 1, dim_);
    // set internal number of vanishing moments
    dtilde2_ = 0;
    m_dtilde2_ = binomialCoefficient(dim_ + dtilde2_ - 1, dim_);
    while (2 * m_dtilde_ > m_dtilde2_) {
      ++dtilde2_;
      m_dtilde2_ = binomialCoefficient(dim_ + dtilde2_ - 1, dim_);
    }
    idcs_.init(dim_, dtilde2_ - 1);
    eigen_assert(idcs_.get_MultiIndexSet().size() == m_dtilde2_ &&
                 "dimension mismatch in total degree indexset");
    IndexType i = 0;
    IndexType j = 0;
    multinomial_coeffs_.resize(m_dtilde2_, m_dtilde2_);
    for (const auto &beta : idcs_.get_MultiIndexSet()) {
      for (const auto &alpha : idcs_.get_MultiIndexSet()) {
        multinomial_coeffs_(j, i) = multinomialCoefficient(alpha, beta);
        ++j;
      }
      ++i;
      j = 0;
    }
    return;
  }

  const eigenMatrix &multinomial_coeffs() const { return multinomial_coeffs_; }
  eigenMatrix shift_matrix(const eigenMatrix &shift) const {
    return internal::momentShifter<Derived, MultiIndexSet>(shift, idcs_,
                                                           multinomial_coeffs_);
  }
  eigenMatrix moment_matrix(const eigenMatrix &P,
                            const ClusterTreeBase<Derived> &CT) const {
    eigenMatrix retval(idcs_.get_MultiIndexSet().size(), CT.indices().size());
    eigenMatrix mp = 0.5 * (CT.bb().col(0) + CT.bb().col(1));
    IndexType i = 0;
    retval.setOnes();
    for (auto j = 0; j < retval.cols(); ++j) {
      i = 0;
      for (const auto &it : idcs_.get_MultiIndexSet()) {
        for (auto k = 0; k < dim_; ++k)
          // make sure that 0^0 = 1
          if (it[k])
            retval(i, j) *= std::pow(P(k, CT.indices()[j]) - mp(k), it[k]);
        ++i;
      }
    }
    return retval;
  }

  IndexType mdtilde() const { return m_dtilde_; }
  IndexType mdtilde2() const { return m_dtilde2_; }

 private:
  IndexType dim_;
  IndexType dtilde_;
  IndexType dtilde2_;
  IndexType m_dtilde_;
  IndexType m_dtilde2_;
  MultiIndexSet idcs_;
  eigenMatrix multinomial_coeffs_;
};

}  // namespace FMCA
#endif

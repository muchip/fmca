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
// This class is borrowd from muchip/SPQR
//
#ifndef FMCA_UTIL_MULTIINDEXSET_H_
#define FMCA_UTIL_MULTIINDEXSET_H_

#include <algorithm>
#include <array>
#include <set>

#include "Macros.h"

namespace FMCA {

enum IndexSetType { TotalDegree, TensorProduct };

template <IndexSetType T> struct IndexSetCriterion {};

template <> struct IndexSetCriterion<TotalDegree> {
  IndexSetCriterion(){};
  IndexSetCriterion(IndexType max_degree) : max_degree_(max_degree) {}
  template <typename T> bool operator()(const T &index) {
    IndexType sum = 0;
    for (auto i : index)
      sum += i;
    return sum <= max_degree_;
  }
  IndexType max_degree_;
};

template <> struct IndexSetCriterion<TensorProduct> {
  IndexSetCriterion(){};
  IndexSetCriterion(IndexType max_degree) : max_degree_(max_degree) {}
  template <typename T> bool operator()(const T &index) {
    IndexType max = 0;
    for (auto i : index)
      max = max > i ? max : i;
    return max <= max_degree_;
  }
  IndexType max_degree_;
};

/**
 *  \brief in order to obtain hierarchies in the index set, we employ
 *  a normwise ordering combined with the lexicographical one
 **/
template <typename Array> struct FMCA_Compare {
  bool operator()(const Array &a, const Array &b) const {
    typename Array::value_type nrma = 0;
    typename Array::value_type nrmb = 0;
    for (auto i = 0; i < a.size(); ++i)
      nrma += std::abs(double(a[i]));
    for (auto i = 0; i < b.size(); ++i)
      nrmb += std::abs(double(b[i]));
    if (nrma != nrmb)
      return nrma < nrmb;
    else
      return std::lexicographical_compare(a.cbegin(), a.cend(), b.cbegin(),
                                          b.cend());
  }
};

/**
 *  \brief specialization for index sets with a general boolean criterion
 **/
template <IndexType Dim, IndexSetType T = TotalDegree> class MultiIndexSet {
public:
  MultiIndexSet(){};
  MultiIndexSet(IndexType max_degree) { init(max_degree); }

  void init(IndexType max_degree) {
    max_degree_ = max_degree;
    is_element_.max_degree_ = max_degree;
    multi_index_set_.clear();
    std::array<IndexType, Dim> index;
    std::fill(index.begin(), index.end(), 0);
    if (is_element_(index)) {
      multi_index_set_.insert(index);
      // compute all other indices in the set recursively
      addChildren(0, index);
    }
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  IndexType max_degree() const { return max_degree; }

  const std::set<std::array<IndexType, Dim>,
                 FMCA_Compare<std::array<IndexType, Dim>>> &
  get_MultiIndexSet() const {
    return multi_index_set_;
  }
  //////////////////////////////////////////////////////////////////////////////
private:
  void addChildren(IndexType max_bit, std::array<IndexType, Dim> &index) {
    // successively increase all entries in the current index
    for (auto i = max_bit; i < Dim; ++i) {
      index[i] += 1;
      if (is_element_(index)) {
        multi_index_set_.insert(index);
        // check child indices only if father index is contained in set.
        // This is sufficient due to the downward closedness assumption
        addChildren(i, index);
      }
      index[i] -= 1;
    }
    return;
  }
  std::set<std::array<IndexType, Dim>, FMCA_Compare<std::array<IndexType, Dim>>>
      multi_index_set_;
  IndexSetCriterion<T> is_element_;
  IndexType max_degree_;
};

} // namespace FMCA

#endif

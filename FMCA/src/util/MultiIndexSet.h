// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU General Public License version 3
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
// This class is borrowd from muchip/SPQR
//
#ifndef FMCA_UTIL_MULTIINDEXSET_H_
#define FMCA_UTIL_MULTIINDEXSET_H_

#include <algorithm>
#include <set>
#include <vector>

#include "Macros.h"

namespace FMCA {

enum IndexSetType { Generic, TotalDegree, TensorProduct, WeightedTotalDegree };

template <IndexSetType T> class MultiIndexSet;

template <IndexSetType T> struct MultiIndexSetInitializer;

template <IndexSetType T> struct IndexSetCriterion {};

template <> struct IndexSetCriterion<TotalDegree> {
  IndexSetCriterion(){};
  IndexSetCriterion(Index max_degree) : max_degree_(max_degree) {}
  template <typename T> bool operator()(const T &index) {
    Index sum = 0;
    for (auto i : index)
      sum += i;
    return sum <= max_degree_;
  }
  Index max_degree_;
};

template <> struct IndexSetCriterion<WeightedTotalDegree> {
  IndexSetCriterion(){};
  IndexSetCriterion(Index max_degree, const std::vector<Scalar> &weights)
      : max_degree_(max_degree), weights_(weights) {}

  template <typename T> bool operator()(const T &index) {
    assert(index.size() == weights_.size() && "dimension mismatch");
    Scalar sum = 0;
    for (auto i = 0; i < index.size(); ++i)
      sum += index[i] * weights_[i];
    return sum <= max_degree_;
  }

  Index max_degree_;
  std::vector<Scalar> weights_;
};

template <> struct IndexSetCriterion<TensorProduct> {
  IndexSetCriterion(){};
  IndexSetCriterion(Index max_degree) : max_degree_(max_degree) {}
  template <typename T> bool operator()(const T &index) {
    Index max = 0;
    for (auto i : index)
      max = max > i ? max : i;
    return max <= max_degree_;
  }
  Index max_degree_;
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
      nrma += std::abs(Scalar(a[i]));
    for (auto i = 0; i < b.size(); ++i)
      nrmb += std::abs(Scalar(b[i]));
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
template <IndexSetType T = TotalDegree> class MultiIndexSet {
  friend struct MultiIndexSetInitializer<T>;

public:
  typedef std::set<std::vector<Index>, FMCA_Compare<std::vector<Index>>>
      multi_index_set;
  MultiIndexSet(){};
  template <typename... Ts> MultiIndexSet(Ts &&...ts) {
    init(std::forward<Ts>(ts)...);
  }

  template <typename... Ts> void init(Ts &&...ts) {
    MultiIndexSetInitializer<T>::init(*this, std::forward<Ts>(ts)...);
  }

  //////////////////////////////////////////////////////////////////////////////
  const Index max_degree() const { return max_degree_; }
  Index &max_degree() { return max_degree_; }

  const Index dim() const { return dim_; }
  Index &dim() { return dim_; }

  const multi_index_set &index_set() const { return s_data_; }
  multi_index_set &index_set() { return s_data_; }

  IndexSetCriterion<T> &is_element() { return is_element_; }
  const IndexSetCriterion<T> &is_element() const { return is_element_; }
  //////////////////////////////////////////////////////////////////////////////
private:
  multi_index_set s_data_;
  IndexSetCriterion<T> is_element_;
  Index max_degree_;
  Index dim_;
};

/**
 *  \brief provide the different initializers for the different index sets
 *
 **/
template <> struct MultiIndexSetInitializer<Generic> {
  template <typename T> static void init(T &set, Index dim, Index max_degree) {
    set.dim() = dim;
    set.max_degree() = max_degree;
    set.is_element().max_degree_ = max_degree;
    set.index_set().clear();
    std::vector<Index> index(set.dim(), 0);
    if (set.is_element()(index)) {
      set.index_set().insert(index);
      // compute all other indices in the set recursively
      addChildren(set, 0, index);
    }
    return;
  }

  template <typename T>
  static void addChildren(T &set, Index max_bit, std::vector<Index> &index) {
    // successively increase all entries in the current index
    for (auto i = max_bit; i < set.dim(); ++i) {
      index[i] += 1;
      if (set.is_element()(index)) {
        set.index_set().insert(index);
        // check child indices only if father index is contained in set.
        // This is sufficient due to the downward closedness assumption
        addChildren(set, i, index);
      }
      index[i] -= 1;
    }
    return;
  }
};

template <>
struct MultiIndexSetInitializer<TotalDegree>
    : public MultiIndexSetInitializer<Generic> {};

template <>
struct MultiIndexSetInitializer<TensorProduct>
    : public MultiIndexSetInitializer<Generic> {};

template <> struct MultiIndexSetInitializer<WeightedTotalDegree> {
  template <typename T>
  static void init(T &set, Index dim, Index max_degree,
                   const std::vector<Scalar> &weights) {
    set.dim() = dim;
    set.max_degree() = max_degree;
    set.is_element().weights_ = weights;
    set.is_element().max_degree_ = max_degree;
    set.index_set().clear();
    std::vector<Index> index(set.dim(), 0);
    if (0 <= set.max_degree()) {
      set.index_set().insert(index);
      // compute all other indices in the set recursively
      addChildren(set, 0, index, set.max_degree());
    }
    return;
  }

  template <typename T>
  static void addChildren(T &set, Index max_bit, std::vector<Index> &index,
                          Scalar q) {
    // successively increase all entries in the current index
    for (auto i = max_bit; i < set.dim(); ++i) {
      // in total degree sets, the remaining budget can be determined by
      // updating the threshold
      index[i] += 1;
      q -= set.is_element().weights_[i];
      if (q >= 0) {
        set.index_set().insert(index);
        // check child indices only if father index is contained in set.
        // This is sufficient due to the downward closedness assumption
        addChildren(set, i, index, q);
        q += set.is_element().weights_[i];
        index[i] -= 1;
      } else {
        q += set.is_element().weights_[i];
        index[i] -= 1;
        break;
      }
    }
    return;
  }
};
} // namespace FMCA

#endif

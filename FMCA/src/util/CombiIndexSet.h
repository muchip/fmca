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
#ifndef FMCA_UTIL_COMBIINDEXSET_H_
#define FMCA_UTIL_COMBIINDEXSET_H_

#include <algorithm>
#include <set>
#include <vector>

#include "Macros.h"
#include "MultiIndexSet.h"
namespace FMCA {

template <IndexSetType T>
class CombiIndexSet;

template <IndexSetType T>
struct CombiIndexSetInitializer;

/**
 *  \brief specialization for index sets with a general boolean criterion
 **/
template <IndexSetType T = TotalDegree>
class CombiIndexSet {
  friend struct CombiIndexSetInitializer<T>;

 public:
  typedef std::map<std::vector<Index>, std::ptrdiff_t,
                   FMCA_Compare<std::vector<Index>>>
      combi_index_set;
  CombiIndexSet() {};
  template <typename... Ts>
  CombiIndexSet(Ts &&...ts) {
    init(std::forward<Ts>(ts)...);
  }

  template <typename... Ts>
  void init(Ts &&...ts) {
    CombiIndexSetInitializer<T>::init(*this, std::forward<Ts>(ts)...);
  }

  //////////////////////////////////////////////////////////////////////////////
  const Index max_degree() const { return max_degree_; }
  Index &max_degree() { return max_degree_; }

  const Index dim() const { return dim_; }
  Index &dim() { return dim_; }

  const combi_index_set &index_set() const { return s_data_; }
  combi_index_set &index_set() { return s_data_; }

  IndexSetCriterion<T> &is_element() { return is_element_; }
  const IndexSetCriterion<T> &is_element() const { return is_element_; }
  //////////////////////////////////////////////////////////////////////////////
 private:
  combi_index_set s_data_;
  IndexSetCriterion<T> is_element_;
  Index max_degree_;
  Index dim_;
};

/**
 *  \brief provide the different initializers for the different index sets
 *
 **/
template <>
struct CombiIndexSetInitializer<Generic> {
  template <typename T>
  static void init(T &set, Index dim, Index max_degree) {
    set.dim() = dim;
    set.max_degree() = max_degree;
    set.is_element().max_degree_ = max_degree;
    set.index_set().clear();
    std::vector<Index> index(set.dim(), 0);
    std::vector<Index> indexP1(set.dim(), 1);
    std::ptrdiff_t cw = 0;
    if (set.is_element()(index)) {
      if (not set.is_element()(indexP1))
        cw = combinationWeight(set, 0, 1, 1, index);
      if (cw) set.index_set().insert(std::make_pair(index, cw));
      // compute all other indices in the set recursively
      addChildren(set, 0, index, indexP1);
    }
    return;
  }

  template <typename T>
  static void addChildren(T &set, Index max_bit, std::vector<Index> &index,
                          std::vector<Index> &indexP1) {
    // successively increase all entries in the current index
    std::ptrdiff_t cw = 0;
    for (auto i = max_bit; i < set.dim(); ++i) {
      index[i] += 1;
      indexP1[i] += 1;
      if (set.is_element()(index)) {
        if (not set.is_element()(indexP1))
          cw = combinationWeight(set, 0, 1, 1, index);
        if (cw) set.index_set().insert(std::make_pair(index, cw));
        // check child indices only if father index is contained in set.
        // This is sufficient due to the downward closedness assumption
        addChildren(set, i, index, indexP1);
      }
      index[i] -= 1;
      indexP1[i] -= 1;
    }
    return;
  }

  template <typename T>
  static std::ptrdiff_t combinationWeight(T &set, Index max_bit,
                                          std::ptrdiff_t cw, Index lvl,
                                          std::vector<Index> &index) {
    // successively check all bits of the multiindex is contained
    for (auto i = max_bit; i < set.dim(); ++i) {
      index[i] += 1;
      if (set.is_element()(index)) {
        if (lvl % 2)
          --cw;
        else
          ++cw;
        // again, exploit downward closedness and perform recursion only
        // if father is contained in Xw_alpha
        cw = combinationWeight(set, i + 1, cw, lvl + 1, index);
      }
      index[i] -= 1;
    }

    return cw;
  }
};

template <>
struct CombiIndexSetInitializer<TotalDegree>
    : public CombiIndexSetInitializer<Generic> {};

template <>
struct CombiIndexSetInitializer<TensorProduct>
    : public CombiIndexSetInitializer<Generic> {};

template <>
struct CombiIndexSetInitializer<WeightedTotalDegree> {
  template <typename T>
  static void init(T &set, Index dim, Index max_degree,
                   const std::vector<Scalar> &weights) {
    set.dim() = dim;
    set.max_degree() = max_degree;
    set.is_element().weights_ = weights;
    set.is_element().max_degree_ = max_degree;
    set.index_set().clear();
    std::vector<Index> index(set.dim(), 0);
    std::ptrdiff_t cw = 0;
    Scalar sumw = 0;
    if (0 <= set.max_degree()) {
      for (auto i = 0; i < dim; ++i) sumw += weights[i];
      if (sumw > set.max_degree())
        cw = combinationWeight(set, 0, 1, 1, set.max_degree());
      if (cw) set.index_set().insert(std::make_pair(index, cw));
      // compute all other indices in the set recursively
      addChildren(set, 0, index, set.max_degree(), sumw);
    }
    return;
  }

  template <typename T>
  static void addChildren(T &set, Index max_bit, std::vector<Index> &index,
                          Scalar q, Scalar sumw) {
    std::ptrdiff_t cw = 0;
    // successively increase all entries in the current index
    for (auto i = max_bit; i < set.dim(); ++i) {
      // in total degree sets, the remaining budget can be determined by
      // updating the threshold
      index[i] += 1;
      q -= set.is_element().weights_[i];
      if (q >= 0) {
        Scalar scap = 0;
        for (auto j = 0; j < set.dim(); ++j)
          scap += Scalar(index[j]) * set.is_element().weights_[j];
        if (scap > set.max_degree() - sumw)
          cw = combinationWeight(set, 0, 1, 1, set.max_degree() - scap);
        if (cw) set.index_set().insert(std::make_pair(index, cw));
        // check child indices only if father index is contained in set.
        // This is sufficient due to the downward closedness assumption
        addChildren(set, i, index, q, sumw);
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

  template <typename T>
  static std::ptrdiff_t combinationWeight(T &set, Index maxBit,
                                          std::ptrdiff_t cw, Index lvl,
                                          Scalar q) {
    for (auto i = maxBit; i < set.dim(); ++i) {
      q -= set.is_element().weights_[i];
      if (q >= 0) {
        if (lvl % 2)
          --cw;
        else
          ++cw;
        cw = combinationWeight(set, i + 1, cw, lvl + 1, q);
        q += set.is_element().weights_[i];
      } else {
        // this is the major difference to the general class, we may break
        // here due to the increasingly ordered weights
        q += set.is_element().weights_[i];
        break;
      }
    }
    return cw;
  }
};

}  // namespace FMCA

#endif

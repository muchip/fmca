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
#ifndef UTIL_MULTIINDEXSET_H_
#define UTIL_MULTIINDEXSET_H_

#include <set>

enum IndexSetType { TotalDegree, TensorProduct };

template <IndexSetType T> struct IndexSetCriterion {};

template <> struct IndexSetCriterion<TotalDegree> {
  IndexSetCriterion(){};
  IndexSetCriterion(unsigned int max_degree) : max_degree_(max_degree) {}
  template <typename T> bool operator()(const T &index) {
    double sum = 0;
    for (auto i : index)
      sum += double(i);
    return sum <= max_degree_;
  }
  unsigned int max_degree_;
};

template <> struct IndexSetCriterion<TensorProduct> {
  IndexSetCriterion(){};
  IndexSetCriterion(unsigned int max_degree) : max_degree_(max_degree) {}
  template <typename T> bool operator()(const T &index) {
    double max = 0;
    for (auto i : index)
      max = max > double(i) ? max : double(i);
    return max <= max_degree_;
  }
  unsigned int max_degree_;
};

/**
 *  \brief in order to obtain hierarchies in the index set, we employ
 *  a normwise ordering combined with the lexicographical one
 **/
template <typename Index> struct FMCA_Compare {
  bool operator()(const Index &a, const Index &b) const {
    double nrma = 0;
    double nrmb = 0;
    for (auto i = 0; i < a.size(); ++i)
      nrma += std::abs(double(a[i]));
    for (auto i = 0; i < b.size(); ++i)
      nrmb += std::abs(double(b[i]));
    if (nrma != nrmb)
      return nrma < nrmb;
    else
      return !std::lexicographical_compare(a.cbegin(), a.cend(), b.cbegin(),
                                           b.cend());
  }
};

/**
 *  \brief specialization for index sets with a general boolean criterion
 **/
template <class Index, IndexSetType T = TotalDegree> class MultiIndexSet {
public:
  MultiIndexSet(){};
  MultiIndexSet(unsigned int dim, unsigned int max_degree) {
    init(dim, max_degree);
  }

  void init(unsigned int dim, unsigned int max_degree) {
    dim_ = dim;
    max_degree_ = max_degree;
    is_element_.max_degree_ = max_degree;
    multi_index_set_.clear();
    Index index;
    index.resize(dim_);
    std::fill(index.begin(), index.end(), 0);
    if (is_element_(index)) {
      multi_index_set_.insert(index);
      // compute all other indices in the set recursively
      addChildren(0, index);
    }
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  unsigned int max_degree() const { return max_degree; }
  unsigned int dim() const { return dim; }

  const std::set<Index, FMCA_Compare<Index>> &get_MultiIndexSet() const {
    return multi_index_set_;
  }
  //////////////////////////////////////////////////////////////////////////////
private:
  void addChildren(unsigned int max_bit, Index &index) {
    // successively increase all entries in the current index
    for (auto i = max_bit; i < dim_; ++i) {
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
  std::set<Index, FMCA_Compare<Index>> multi_index_set_;
  IndexSetCriterion<T> is_element_;
  unsigned int max_degree_;
  unsigned int dim_;
};

#endif

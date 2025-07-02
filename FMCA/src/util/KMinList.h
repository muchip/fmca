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
#ifndef FMCA_UTIL_KMINLIST_H_
#define FMCA_UTIL_KMINLIST_H_

#include <set>

#include "Macros.h"

namespace FMCA {

class KMinList {
 public:
  using value_type = std::pair<Index, Scalar>;

  struct cmp {
    bool operator()(const value_type& a, const value_type& b) const {
      return a.second < b.second;
    }
  };
  using PriorityQueue =
      std::priority_queue<value_type, std::vector<value_type>, cmp>;

  template <typename T>
  typename T::container_type& container_getter(T& t) {
    struct hack : private T {
      static typename T::container_type& get(T& t) { return t.*&hack::c; }
    };
    return hack::get(t);
  }

  KMinList() noexcept : k_(1) {}
  KMinList(Index k) noexcept : k_(k) {}

  void insert(const value_type& tuple) {
    if (queue_.size() < k_) {
      queue_.push(tuple);
    } else if (tuple.second < queue_.top().second) {
      queue_.pop();
      queue_.push(tuple);
    }
  }

  Scalar max_min() const {
    if (queue_.empty()) return FMCA_INF;
    return queue_.top().second;
  }

  const std::vector<value_type>& list() { return container_getter(queue_); }

 private:
  PriorityQueue queue_;
  Index k_;
};

}  // namespace FMCA
#endif

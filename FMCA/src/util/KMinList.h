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
  struct my_less {
    bool operator()(const value_type &a, const value_type &b) const {
      return a.second == b.second ? a.first < b.first : a.second < b.second;
    }
  };

  KMinList() noexcept { k_ = 0; }
  KMinList(Index k) noexcept { k_ = k; }

  void insert(const value_type &tuple) {
    if (queue_.size() < k_)
      queue_.emplace(tuple);
    else {
      if (tuple.second < queue_.rbegin()->second) {
        std::set<value_type, my_less>::iterator it = queue_.find(tuple);
        if (it == queue_.end()) {
          queue_.erase(std::prev(it));
          queue_.emplace(tuple);
        }
      }
    }
  }

  bool isFull() const { return queue_.size() == k_; }

  const std::set<value_type, my_less> &list() const { return queue_; }

 private:
  std::set<value_type, my_less> queue_;
  Index k_;
};
}  // namespace FMCA
#endif

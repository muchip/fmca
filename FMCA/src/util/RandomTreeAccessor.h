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
#ifndef FMCA_UTIL_RANDOMTREEACCESSOR_H_
#define FMCA_UTIL_RANDOMTREEACCESSOR_H_

#include <vector>

#include "TreeBase.h"

namespace FMCA {
namespace internal {
template <typename Derived> class RandomTreeAccessor {
public:
  RandomTreeAccessor(){};
  RandomTreeAccessor(const TreeBase<Derived> &T, const Index res_mem = 1000) {
    init(T, res_mem);
  };

  void init(const TreeBase<Derived> &T, const Index res_mem = 1000) {
    Index cur_level = 0;
    nodes_.reserve(res_mem);
    levels_.reserve(res_mem);
    max_level_ = 0;
    levels_.push_back(0);
    for (const auto &it : T) {
      if (it.level() != cur_level) {
        levels_.push_back(nodes_.size());
        ++cur_level;
      }
      nodes_.push_back(std::addressof(it.derived()));
      max_level_ = it.level() > max_level_ ? it.level() : max_level_;
    }
    levels_.push_back(nodes_.size());
    nodes_.shrink_to_fit();
    levels_.shrink_to_fit();
  }

  Index max_level() const { return max_level_; }
  const std::vector<const Derived *> &nodes() const { return nodes_; }
  const std::vector<Index> &levels() const { return levels_; }

private:
  std::vector<const Derived *> nodes_;
  std::vector<Index> levels_;
  Index max_level_;
};
} // namespace internal
} // namespace FMCA
#endif

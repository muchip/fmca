// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_MEMORYARENA__
#define FMCA_UTIL_MEMORYARENA__

#include "Macros.h"

namespace FMCA {
template <typename T>
class MemoryArena {
 public:
  MemoryArena() : slab_size_(0), slabs_in_use_(0), adopted_(0) {}

  explicit MemoryArena(Index slab_size, Index initial_capacity = 64)
      : slab_size_(slab_size), slabs_in_use_(0), adopted_(0) {
    free_list_.reserve(initial_capacity);
  }

  void init(Index slab_size, Index initial_capacity = 64) {
    slab_size_ = slab_size;
    slabs_in_use_ = 0;
    adopted_ = 0;
    free_list_.reserve(initial_capacity);
  }

  std::unique_ptr<T[]> acquire() {
    if (!free_list_.empty()) {
      std::unique_ptr<T[]> slab = std::move(free_list_.back());
      free_list_.pop_back();
      // if we're consuming a previously adopted slab, it's now
      // "in use" under this arena's own accounting again
      if (adopted_ > 0)
        --adopted_;
      else
        ++slabs_in_use_;
      return slab;
    }
    ++slabs_in_use_;
    return std::make_unique<T[]>(slab_size_);
  }

  /// return a slab to the free list. If this arena did not issue it
  /// (slabs_in_use_ would go negative), track it as adopted instead.
  void release(std::unique_ptr<T[]> slab) {
    free_list_.push_back(std::move(slab));
    if (slabs_in_use_ > 0)
      --slabs_in_use_;
    else
      ++adopted_;
  }

  Index slab_size() const { return slab_size_; }
  Index slabs_in_use() const { return slabs_in_use_; }
  Index adopted() const { return adopted_; }
  Index num_free_slabs() const { return static_cast<Index>(free_list_.size()); }
  Index capacity() const { return static_cast<Index>(free_list_.capacity()); }

 private:
  Index slab_size_;
  Index slabs_in_use_;
  Index adopted_;
  std::vector<std::unique_ptr<T[]>> free_list_;
};
}  // namespace FMCA
#endif

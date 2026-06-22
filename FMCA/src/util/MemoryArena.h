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
  //////////////////////////////////////////////////////////////////////////////
  MemoryArena() : slab_size_(0), slabs_in_use_(0) {}

  explicit MemoryArena(Index slab_size, Index initial_capacity = 64)
      : slab_size_(slab_size), slabs_in_use_(0) {
    free_list_.reserve(initial_capacity);
  }

  MemoryArena(const MemoryArena &) = delete;
  MemoryArena(MemoryArena &&) = delete;
  //////////////////////////////////////////////////////////////////////////////
  /// acquire a slab from the free list, allocating a new one if necessary.
  std::unique_ptr<T[]> acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++slabs_in_use_;
    if (free_list_.empty())
      return std::make_unique<T[]>(slab_size_);
    else {
      std::unique_ptr<T[]> slab = std::move(free_list_.back());
      free_list_.pop_back();
      return slab;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  /// return a slab to the free list for reuse.
  void release(std::unique_ptr<T[]> slab) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push_back(std::move(slab));
    --slabs_in_use_;
  }

  //////////////////////////////////////////////////////////////////////////////
  Index slab_size() const { return slab_size_; }

  Index slabs_in_use() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slabs_in_use_;
  }

  Index num_free_slabs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<Index>(free_list_.size());
  }

  Index capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<Index>(free_list_.capacity());
  }

 private:
  const Index slab_size_;
  Index slabs_in_use_;
  std::vector<std::unique_ptr<T[]>> free_list_;
  mutable std::mutex mutex_;
};
}  // namespace FMCA
#endif

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
  MemoryArena(const MemoryArena &) = delete;
  MemoryArena(MemoryArena &&) = delete;
  struct Deleter {
    void operator()(T *p) const { Eigen::internal::aligned_free(p); }
  };
  struct Allocator {
    T *operator()(Index n) const {
      return static_cast<T *>(Eigen::internal::aligned_malloc(n * sizeof(T)));
    }
  };

  using Ptr = std::unique_ptr<T[], Deleter>;

  static constexpr Index aligned_stride(Index n) {
    const Index elems = n * n;
    const Index bytes_per_chunk = elems * sizeof(T);
    const Index align = alignment_bytes();
    const Index padded_bytes = ((bytes_per_chunk + align - 1) / align) * align;
    return padded_bytes / sizeof(T);
  }

  static Ptr make_ptr(Index n) { return Ptr(Allocator{}(n)); }

  MemoryArena() {}

  MemoryArena(Index slab_size, Index num_stacks = 1,
              Index initial_capacity = 64) {
    init(slab_size, num_stacks, initial_capacity);
  }

  void init(Index slab_size, Index num_stacks = 1,
            Index initial_capacity = 64) {
    slab_size_ = slab_size;
    overflow_threshold_ = 2 * initial_capacity;
    central_.clear();
    stacks_.clear();
    stacks_.resize(num_stacks);
    for (std::vector<Ptr> &stack : stacks_) {
      stack.reserve(initial_capacity);
      for (Index i = 0; i < initial_capacity; ++i)
        stack.emplace_back(make_ptr(slab_size_));
    }
  }

  Ptr acquire(Index tid = 0) {
    std::vector<Ptr> &stack = stacks_[tid];
    if (!stack.empty()) {
      Ptr slab = std::move(stack.back());
      stack.pop_back();
      return slab;
    }
    {
      SpinLock slock(lock_);
      if (!central_.empty()) {
        Ptr slab = std::move(central_.back());
        central_.pop_back();
        return slab;
      }
    }
    return make_ptr(slab_size_);
  }

  void release(Ptr slab, Index tid = 0) {
    std::vector<Ptr> &stack = stacks_[tid];
    stack.push_back(std::move(slab));
    if (stack.size() > overflow_threshold_) {
      SpinLock slock(lock_);
      central_.push_back(std::move(stack.back()));
      stack.pop_back();
    }
    return;
  }

  Index num_free_slabs(Index tid = 0) const { return stacks_[tid].size(); }

 private:
  static constexpr Index alignment_bytes() {
#ifdef EIGEN_MAX_ALIGN_BYTES
    return static_cast<Index>(EIGEN_MAX_ALIGN_BYTES);
#else
    return static_cast<Index>(alignof(T));
#endif
  }

  struct SpinLock {
    explicit SpinLock(std::atomic_flag &f) : f_(f) {
      while (f_.test_and_set(std::memory_order_acquire)) {
      }
    }
    ~SpinLock() { f_.clear(std::memory_order_release); }
    std::atomic_flag &f_;
  };

  Index slab_size_;
  Index overflow_threshold_;
  std::vector<std::vector<Ptr>> stacks_;
  std::vector<Ptr> central_;
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};
}  // namespace FMCA
#endif

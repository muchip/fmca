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
#ifndef FMCA_UTIL_MEMORYPOOL_H_
#define FMCA_UTIL_MEMORYPOOL_H_

#include "Macros.h"

namespace FMCA {

template <typename T>
class MemoryPool {
 public:
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;

  static constexpr Index kBulk = 64;
  static constexpr Index kLocalMax = 512;
  static constexpr Index kChunkBlocks = 256;

  struct AlignedAllocator {
    T *operator()(Index n) const {
      return static_cast<T *>(Eigen::internal::aligned_malloc(n * sizeof(T)));
    }
  };
  struct AlignedDeleter {
    void operator()(T *p) const { Eigen::internal::aligned_free(p); }
  };
  using ChunkPtr = std::unique_ptr<T[], AlignedDeleter>;

  static constexpr Index alignment_bytes() {
#ifdef EIGEN_MAX_ALIGN_BYTES
    return static_cast<Index>(EIGEN_MAX_ALIGN_BYTES);
#else
    return static_cast<Index>(alignof(T));
#endif
  }

  static constexpr Index aligned_stride(Index n) {
    const Index elems = n * n;
    const Index bytes_per_chunk = elems * sizeof(T);
    const Index align = alignment_bytes();
    const Index padded_bytes = ((bytes_per_chunk + align - 1) / align) * align;
    return padded_bytes / sizeof(T);
  }

  MemoryPool() {}

  MemoryPool(Index n, Index nthreads, Index chunk_blocks = kChunkBlocks) {
    init(n, nthreads, chunk_blocks);
  }

  void init(Index n, Index nthreads, Index chunk_blocks = kChunkBlocks) {
    clear();
    block_elems_ = aligned_stride(n);
    chunk_blocks_ = chunk_blocks;
    locals_.resize(nthreads);
    chunks_.resize(nthreads);
    bumps_.assign(nthreads, nullptr);
    remaining_.assign(nthreads, 0);
    for (std::vector<T *> &l : locals_) l.reserve(kLocalMax);
  }

  T *acquire(Index tid = 0) {
    eigen_assert(tid < locals_.size());
    if (!locals_[tid].empty()) {
      T *p = locals_[tid].back();
      locals_[tid].pop_back();
      return p;
    }
    refill(tid);
    if (!locals_[tid].empty()) {
      T *p = locals_[tid].back();
      locals_[tid].pop_back();
      return p;
    }
    return bump_alloc(tid);
  }

  void release(T *p, Index tid = 0) {
    eigen_assert(tid < locals_.size());
    locals_[tid].push_back(p);
    if (locals_[tid].size() >= kLocalMax) spill(tid);
  }

  Index block_elems() const { return block_elems_; }

  void clear() {
    locals_.clear();
    chunks_.clear();  // unique_ptr destructors call aligned_free
    bumps_.clear();
    remaining_.clear();
    central_.clear();
    block_elems_ = 0;
  }

 private:
  T *bump_alloc(Index tid) {
    if (remaining_[tid] == 0) {
      T *raw = AlignedAllocator{}(chunk_blocks_ * block_elems_);
      chunks_[tid].emplace_back(raw);
      bumps_[tid] = raw;
      remaining_[tid] = chunk_blocks_;
    }
    T *p = bumps_[tid];
    bumps_[tid] += block_elems_;
    --remaining_[tid];
    return p;
  }

  void refill(Index tid) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Index n = std::min<Index>(kBulk, central_.size());
    if (n == 0) return;
    locals_[tid].insert(locals_[tid].end(), central_.end() - n, central_.end());
    central_.resize(central_.size() - n);
  }

  void spill(Index tid) {
    const Index n = std::min<Index>(kBulk, locals_[tid].size());
    std::lock_guard<std::mutex> lock(mutex_);
    central_.insert(central_.end(), locals_[tid].end() - n, locals_[tid].end());
    locals_[tid].resize(locals_[tid].size() - n);
  }

  Index block_elems_ = 0;
  Index chunk_blocks_ = kChunkBlocks;
  std::vector<std::vector<T *>> locals_;
  std::vector<std::vector<ChunkPtr>> chunks_;
  std::vector<T *> bumps_;
  std::vector<Index> remaining_;
  std::vector<T *> central_;
  mutable std::mutex mutex_;
};
}  // namespace FMCA
#endif

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
  static constexpr Index kMaxAllocMB = 64;

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

  MemoryPool(Index max_n, Index nthreads, Index max_alloc_mb = kMaxAllocMB) {
    init(max_n, nthreads, max_alloc_mb);
  }

  void init(Index max_n, Index nthreads, Index max_alloc_mb = kMaxAllocMB) {
    clear();
    make_class_sizes(max_n);
    chunk_blocks_ = blocks_per_chunk(max_alloc_mb);
    locals_.resize(nthreads);
    chunks_.resize(nthreads);
    bumps_.assign(nthreads, nullptr);
    remaining_.assign(nthreads, 0);
    central_.resize(class_sizes_.size());
    for (std::vector<std::vector<T *>> &t : locals_) {
      t.resize(class_sizes_.size());
      for (std::vector<T *> &l : t) l.reserve(kLocalMax);
    }
  }

  T *acquire(Index n, Index tid = 0) {
    eigen_assert(tid < locals_.size());
    const Index cls = class_of(aligned_stride(n));
    std::vector<T *> &local = locals_[tid][cls];
    if (!local.empty()) {
      T *p = local.back();
      local.pop_back();
      return p;
    }
    refill(tid, cls);
    if (!local.empty()) {
      T *p = local.back();
      local.pop_back();
      return p;
    }
    return bump_alloc(tid, cls);
  }

  void release(T *p, Index n, Index tid = 0) {
    eigen_assert(tid < locals_.size());
    const Index cls = class_of(aligned_stride(n));
    std::vector<T *> &local = locals_[tid][cls];
    local.push_back(p);
    if (local.size() >= kLocalMax) spill(tid, cls);
  }

  Index block_elems(Index n) const { return class_sizes_[class_of(aligned_stride(n))]; }

  void clear() {
    locals_.clear();
    chunks_.clear();  // unique_ptr destructors call aligned_free
    bumps_.clear();
    remaining_.clear();
    central_.clear();
    class_sizes_.clear();
    chunk_blocks_ = 0;
  }

 private:
  // Power-of-two ladder from align_elems() up to the next power of two
  // covering max_n. Every class size is thus a multiple of align_elems(),
  // so every bump-carved block is automatically Eigen-aligned.
  void make_class_sizes(Index max_n) {
    const Index align = alignment_bytes();
    const Index elem_bytes = static_cast<Index>(sizeof(T));
    Index g_a = align, g_b = elem_bytes;
    while (g_b != 0) {
      const Index t = g_a % g_b;
      g_a = g_b;
      g_b = t;
    }
    const Index align_elems = align / g_a;
    Index hi = 1;
    const Index needed = aligned_stride(max_n);
    while (hi < needed) hi <<= 1;
    for (Index c = align_elems; c < hi; c <<= 1) class_sizes_.push_back(c);
    class_sizes_.push_back(hi);
  }

  // Plain integer scan over the small class_sizes_ list: no transcendental
  // functions, no bit-count magic numbers, exact for any monotone ladder.
  Index class_of(Index elems) const {
    Index i = 0;
    const Index last = static_cast<Index>(class_sizes_.size()) - 1;
    while (i < last && class_sizes_[i] < elems) ++i;
    return i;
  }

  // Same one-time bulk/hysteresis semantics as the baseline, just indexed
  // per (thread, class) instead of a single flat list.
  void refill(Index tid, Index cls) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<T *> &cen = central_[cls];
    const Index n = std::min<Index>(kBulk, cen.size());
    if (n == 0) return;
    locals_[tid][cls].insert(locals_[tid][cls].end(), cen.end() - n, cen.end());
    cen.resize(cen.size() - n);
  }

  void spill(Index tid, Index cls) {
    std::vector<T *> &local = locals_[tid][cls];
    const Index n = std::min<Index>(kBulk, local.size());
    std::lock_guard<std::mutex> lock(mutex_);
    central_[cls].insert(central_[cls].end(), local.end() - n, local.end());
    local.resize(local.size() - n);
  }

  // chunk_blocks_ counts largest-class blocks per chunk, sized from
  // kMaxAllocMB (or a caller-supplied override), matching the spirit of
  // the baseline's kChunkBlocks but derived from a memory budget instead
  // of a fixed block count.
  Index blocks_per_chunk(Index max_alloc_mb) const {
    const Index mb = Index(1024) * Index(1024);
    const Index bytes_per_block = class_sizes_.back() * static_cast<Index>(sizeof(T));
    Index bytes_budget = max_alloc_mb * mb;
    if (bytes_budget < bytes_per_block) {
      const Index min_mb = (bytes_per_block + mb - 1) / mb;
      std::cerr << "MemoryPool: max_alloc_mb=" << max_alloc_mb
                 << " too small for one largest-class block; rounding up to "
                 << min_mb << " MB.\n";
      bytes_budget = min_mb * mb;
    }
    const Index blocks = bytes_budget / bytes_per_block;
    return blocks > 0 ? blocks : 1;
  }

  // Kept structurally identical to the baseline's bump_alloc: one raw
  // aligned_malloc per chunk, bump pointer advances by the requested
  // class's block size, remaining_ counts blocks (of the LARGEST class'
  // worth of elements) left in the current chunk.
  T *bump_alloc(Index tid, Index cls) {
    if (remaining_[tid] == 0) {
      T *raw = AlignedAllocator{}(chunk_blocks_ * class_sizes_.back());
      chunks_[tid].emplace_back(raw);
      bumps_[tid] = raw;
      remaining_[tid] = chunk_blocks_ * class_sizes_.back();
    }
    const Index sz = class_sizes_[cls];
    eigen_assert(sz <= remaining_[tid]);
    T *p = bumps_[tid];
    bumps_[tid] += sz;
    remaining_[tid] -= sz;
    return p;
  }

  std::vector<Index> class_sizes_;
  Index chunk_blocks_ = 0;
  std::vector<std::vector<std::vector<T *>>> locals_;
  std::vector<std::vector<ChunkPtr>> chunks_;
  std::vector<T *> bumps_;
  std::vector<Index> remaining_;
  std::vector<std::vector<T *>> central_;
  mutable std::mutex mutex_;
};
}  // namespace FMCA
#endif  // FMCA_UTIL_MEMORYPOOL_H_


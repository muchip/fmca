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

  static constexpr Index aligned_stride(Index n) {
    const Index elems = n * n;
    const Index bytes_per_chunk = elems * sizeof(T);
    const Index align = alignment_bytes();
    const Index padded_bytes = ((bytes_per_chunk + align - 1) / align) * align;
    return padded_bytes / sizeof(T);
  }

  static Index align_elems() {
    Index a = alignment_bytes();
    Index b = static_cast<Index>(sizeof(T));
    while (b != 0) {
      const Index t = a % b;
      a = b;
      b = t;
    }
    return alignment_bytes() / a;
  }

  inline Index align_up(Index elems) const {
    return ((elems + align_elems_ - 1) / align_elems_) * align_elems_;
  }

  MemoryPool() {}

  MemoryPool(Index max_elems, Index nthreads,
             Index max_alloc_mb = kMaxAllocMB) {
    init(max_elems, nthreads, max_alloc_mb);
  }

  void init(Index max_elems, Index nthreads, Index max_alloc_mb = kMaxAllocMB) {
    clear();
    align_elems_ = align_elems();
    make_class_sizes(max_elems);
    chunk_blocks_ = blocks_per_chunk(max_alloc_mb);
    //
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

  T *acquire(Index elems, Index tid = 0) {
    const Index cls = class_of(align_up(elems));
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

  void release(T *p, Index elems, Index tid = 0) {
    const Index cls = class_of(align_up(elems));
    std::vector<T *> &local = locals_[tid][cls];
    local.push_back(p);
    if (local.size() >= kLocalMax) spill(tid, cls);
  }

  void clear() {
    locals_.clear();
    chunks_.clear();
    bumps_.clear();
    remaining_.clear();
    central_.clear();
    class_sizes_.clear();
    chunk_blocks_ = 0;
  }

 private:
  struct AlignedAllocator {
    T *operator()(Index n) const {
      return static_cast<T *>(Eigen::internal::aligned_malloc(n * sizeof(T)));
    }
  };
  struct AlignedDeleter {
    void operator()(T *p) const { Eigen::internal::aligned_free(p); }
  };
  using ChunkPtr = std::unique_ptr<T[], AlignedDeleter>;

  inline static constexpr Index alignment_bytes() {
#ifdef EIGEN_MAX_ALIGN_BYTES
    return static_cast<Index>(EIGEN_MAX_ALIGN_BYTES);
#else
    return static_cast<Index>(alignof(T));
#endif
  }

  void make_class_sizes(Index max_elems) {
    const Index needed = align_up(max_elems);
    const Index align_elems2 = align_elems_ * align_elems_;
    Index hi = 1;
    while (hi < needed) hi <<= 1;
    // here is potential space for improvement due to granularity for small
    // allocs
    for (Index c = align_elems2; c < hi; c <<= 1) class_sizes_.push_back(c);
    class_sizes_.push_back(hi);
  }

  Index class_of(Index elems) const {
    const Index lo = class_sizes_.front();
    const Index ratio = (elems + lo - 1) / lo;
    if (ratio <= 1) return 0;
    Index x = ratio - 1;
    Index i = 0;
    while (x != 0) {
      x >>= 1;
      ++i;
    }
    return i;
  }

  Index blocks_per_chunk(Index max_alloc_mb) const {
    constexpr Index mb = 1048576;
    const Index bytes_per_block = class_sizes_.back() * sizeof(T);
    const Index min_mb = (bytes_per_block + mb - 1) / mb;
    if (max_alloc_mb < min_mb) {
      std::cerr << "MemoryPool: max_alloc_mb=" << max_alloc_mb
                << " too small for one largest-class block; rounding up to "
                << min_mb << " MB.\n";
      max_alloc_mb = min_mb;
    }
    return (max_alloc_mb * mb) / bytes_per_block;
  }

  void refill(Index tid, Index cls) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<T *> &cen = central_[cls];
    const Index n = std::min<Index>(Index(kBulk), cen.size());
    if (n == 0) return;
    locals_[tid][cls].insert(locals_[tid][cls].end(), cen.end() - n, cen.end());
    cen.resize(cen.size() - n);
  }

  void spill(Index tid, Index cls) {
    std::vector<T *> &local = locals_[tid][cls];
    const Index n = std::min<Index>(Index(kBulk), local.size());
    std::lock_guard<std::mutex> lock(mutex_);
    central_[cls].insert(central_[cls].end(), local.end() - n, local.end());
    local.resize(local.size() - n);
  }

  T *bump_alloc(Index tid, Index cls) {
    const Index sz = class_sizes_[cls];
    if (remaining_[tid] < sz) {
      T *raw = AlignedAllocator{}(chunk_blocks_ * class_sizes_.back());
      chunks_[tid].emplace_back(raw);
      bumps_[tid] = raw;
      remaining_[tid] = chunk_blocks_ * class_sizes_.back();
    }
    T *p = bumps_[tid];
    bumps_[tid] += sz;
    remaining_[tid] -= sz;
    return p;
  }

  std::vector<Index> class_sizes_;
  std::vector<std::vector<std::vector<T *>>> locals_;
  std::vector<std::vector<ChunkPtr>> chunks_;
  std::vector<T *> bumps_;
  std::vector<Index> remaining_;
  std::vector<std::vector<T *>> central_;
  Index chunk_blocks_ = 0;
  Index align_elems_ = 0;
  mutable std::mutex mutex_;
};

}  // namespace FMCA
#endif

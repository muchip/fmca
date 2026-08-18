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

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "Macros.h"

namespace FMCA {

template <typename T>
class MemoryPool {
 public:
  typedef std::ptrdiff_t difference_type;

  MemoryPool(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  MemoryPool &operator=(MemoryPool &&) = delete;

  static constexpr difference_type kBulk = 64;
  static constexpr difference_type kLocalMax = 512;
  static constexpr difference_type kMaxAllocMB = 64;

  MemoryPool() {}

  MemoryPool(difference_type max_elems, Index nthreads,
             difference_type max_alloc_mb = kMaxAllocMB) {
    init(max_elems, nthreads, max_alloc_mb);
  }

  void init(difference_type max_elems, Index nthreads,
            difference_type max_alloc_mb = kMaxAllocMB) {
    clear();
    align_elems_ = align_elems();
    make_class_sizes(max_elems);
    chunk_blocks_ = blocks_per_chunk(max_alloc_mb);
    const std::size_t ncls = class_sizes_.size();
    locals_.resize(nthreads);
    chunks_.resize(nthreads);
    bumps_.assign(nthreads, nullptr);
    remaining_.assign(nthreads, 0);
    central_.resize(ncls);
    std::vector<std::mutex>(ncls).swap(class_mutex_);
    std::vector<std::atomic<difference_type>>(ncls).swap(central_count_);
    for (std::size_t c = 0; c < ncls; ++c)
      central_count_[c].store(0, std::memory_order_relaxed);
    for (std::vector<std::vector<T *>> &t : locals_) {
      t.resize(ncls);
      for (std::vector<T *> &l : t) l.reserve(kLocalMax);
    }
  }

  T *acquire(difference_type elems, Index tid = 0) {
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

  void release(T *p, difference_type elems, Index tid = 0) {
    const Index cls = class_of(align_up(elems));
    std::vector<T *> &local = locals_[tid][cls];
    local.push_back(p);
    if (static_cast<difference_type>(local.size()) >= kLocalMax)
      spill(tid, cls);
  }

 private:
  void clear() {
    locals_.clear();
    chunks_.clear();
    bumps_.clear();
    remaining_.clear();
    central_.clear();
    class_sizes_.clear();
    std::vector<std::mutex>().swap(class_mutex_);
    std::vector<std::atomic<difference_type>>().swap(central_count_);
    chunk_blocks_ = 0;
    align_elems_ = 0;
  }

  struct AlignedAllocator {
    T *operator()(difference_type n) const {
      return static_cast<T *>(
          Eigen::internal::aligned_malloc(n * difference_type(sizeof(T))));
    }
  };
  struct AlignedDeleter {
    void operator()(T *p) const { Eigen::internal::aligned_free(p); }
  };
  using ChunkPtr = std::unique_ptr<T[], AlignedDeleter>;

  inline static constexpr difference_type alignment_bytes() {
#ifdef EIGEN_MAX_ALIGN_BYTES
    return static_cast<difference_type>(EIGEN_MAX_ALIGN_BYTES);
#else
    return static_cast<difference_type>(alignof(T));
#endif
  }

  static difference_type align_elems() {
    difference_type a = alignment_bytes();
    difference_type b = static_cast<difference_type>(sizeof(T));
    while (b != 0) {
      const difference_type t = a % b;
      a = b;
      b = t;
    }
    return alignment_bytes() / a;
  }

  inline difference_type align_up(difference_type elems) const {
    return ((elems + align_elems_ - 1) / align_elems_) * align_elems_;
  }

  void make_class_sizes(difference_type max_elems) {
    const difference_type needed = align_up(max_elems);
    const difference_type align_elems2 = align_elems_ * align_elems_;
    difference_type hi = 1;
    while (hi < needed) hi <<= 1;
    for (difference_type c = align_elems2; c < hi; c <<= 1)
      class_sizes_.push_back(c);
    class_sizes_.push_back(hi);
  }

  Index class_of(difference_type elems) const {
    const difference_type lo = class_sizes_.front();
    const difference_type ratio = (elems + lo - 1) / lo;
    if (ratio <= 1) return 0;
    difference_type x = ratio - 1;
    Index i = 0;
    while (x != 0) {
      x >>= 1;
      ++i;
    }
    return i;
  }

  difference_type blocks_per_chunk(difference_type max_alloc_mb) const {
    constexpr difference_type mb = 1048576;
    const difference_type bytes_per_block =
        class_sizes_.back() * difference_type(sizeof(T));
    const difference_type min_mb = (bytes_per_block + mb - 1) / mb;
    if (max_alloc_mb < min_mb) {
      std::cerr << "MemoryPool: max_alloc_mb=" << max_alloc_mb
                << " too small for one largest-class block; rounding up to "
                << min_mb << " MB.\n";
      max_alloc_mb = min_mb;
    }
    return (max_alloc_mb * mb) / bytes_per_block;
  }

  void refill(Index tid, Index cls) {
    if (!central_count_[cls].load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(class_mutex_[cls]);
    std::vector<T *> &cen = central_[cls];
    const difference_type n = std::min<difference_type>(kBulk, cen.size());
    if (n == 0) return;
    locals_[tid][cls].insert(locals_[tid][cls].end(), cen.end() - n, cen.end());
    cen.resize(cen.size() - n);
    central_count_[cls].store(static_cast<difference_type>(cen.size()),
                              std::memory_order_release);
  }

  void spill(Index tid, Index cls) {
    std::vector<T *> &local = locals_[tid][cls];
    const difference_type n = std::min<difference_type>(kBulk, local.size());
    std::lock_guard<std::mutex> lock(class_mutex_[cls]);
    std::vector<T *> &cen = central_[cls];
    cen.insert(cen.end(), local.end() - n, local.end());
    local.resize(local.size() - n);
    central_count_[cls].store(static_cast<difference_type>(cen.size()),
                              std::memory_order_release);
  }

  T *bump_alloc(Index tid, Index cls) {
    const difference_type sz = class_sizes_[cls];
    if (remaining_[tid] < sz) {
      const difference_type chunk_elems = chunk_blocks_ * class_sizes_.back();
      T *raw = AlignedAllocator{}(chunk_elems);
      chunks_[tid].emplace_back(raw);
      bumps_[tid] = raw;
      remaining_[tid] = chunk_elems;
    }
    T *p = bumps_[tid];
    bumps_[tid] += sz;
    remaining_[tid] -= sz;
    return p;
  }

  std::vector<difference_type> class_sizes_;
  std::vector<std::vector<std::vector<T *>>> locals_;
  std::vector<std::vector<ChunkPtr>> chunks_;
  std::vector<T *> bumps_;
  std::vector<difference_type> remaining_;
  std::vector<std::vector<T *>> central_;
  std::vector<std::mutex> class_mutex_;
  std::vector<std::atomic<difference_type>> central_count_;
  difference_type chunk_blocks_ = 0;
  difference_type align_elems_ = 0;
};

}  // namespace FMCA
#endif

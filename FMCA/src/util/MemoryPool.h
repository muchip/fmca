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

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace FMCA {

/**
 *  \brief thread-cached, size-class memory pool for matrix blocks
 *
 *  Hands out buffers for single matrices (map them with an aligned
 *  Eigen::Map). Blocks may live arbitrarily long and may be freed by a
 *  different thread than the one that allocated them, so the pool is
 *  agnostic to the traversal order (level-by-level or DAG).
 *
 *  Invariants:
 *  - ALIGNMENT: chunks are allocated with alignment kAlign and every
 *    size class is a multiple of kAlign (static_assert below), hence
 *    every pointer returned by alloc() is kAlign-aligned. kAlign = 128
 *    subsumes any EIGEN_MAX_ALIGN_BYTES, so Eigen::Aligned maps are
 *    always valid, and no two blocks share a cache line.
 *  - POINTER STABILITY: memory is carved from independent fixed-size
 *    chunks that are never reallocated or moved; growing the pool adds
 *    new chunks and never invalidates outstanding pointers.
 *  - THREADS: alloc/free resolve the calling thread's cache internally
 *    (thread_local registration, no thread id in the interface). Any
 *    thread may free any block: it simply lands in the freeing thread's
 *    cache. This remains correct under OpenMP task scheduling and work
 *    stealing, since the executing thread is determined inside the
 *    call, where no task switch can occur.
 *  - free(p, n) must receive the same n as alloc(n); clear()/init()
 *    require quiescence (no concurrent alloc/free, all blocks dead).
 *
 *  Steady-state alloc/free is a push/pop on an intrusive thread-local
 *  freelist (the freed block's first bytes store the next pointer).
 *  Refills/spills move kBulk/kLocalMax/2 blocks per central-lock
 *  acquisition. Nothing is returned to the OS before clear().
 **/
template <typename T>
class MemoryPool {
 public:
  static constexpr std::size_t kAlign = 128;
  static constexpr std::size_t kChunkBytes = std::size_t(8) << 20;
  static constexpr std::size_t kMaxClassBytes = std::size_t(1) << 20;
  static constexpr std::size_t kBulk = 64;
  static constexpr std::size_t kLocalMax = 512;
  static constexpr int kNumClasses = 26;

  // powers of two and halfway steps; all entries multiples of kAlign
  static constexpr std::array<std::size_t, kNumClasses> make_class_sizes() {
    std::array<std::size_t, kNumClasses> s{};
    s[0] = kAlign;
    std::size_t v = 2 * kAlign;
    int i = 1;
    while (i < kNumClasses) {
      s[i++] = v;
      if (i < kNumClasses) s[i++] = v + v / 2;
      v *= 2;
    }
    return s;
  }
  static constexpr std::array<std::size_t, kNumClasses> kClassSizes =
      make_class_sizes();

  static constexpr bool classes_are_aligned() {
    for (std::size_t s : kClassSizes)
      if (s % kAlign) return false;
    return true;
  }
  static_assert(classes_are_aligned(),
                "every size class must be a multiple of kAlign, otherwise "
                "bump carving breaks the alignment guarantee");

  static int class_of(std::size_t bytes) {
    int c = 0;
    while (kClassSizes[c] < bytes) ++c;
    return c;
  }

  MemoryPool() : epoch_(next_epoch()) {}
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  ~MemoryPool() { clear(); }

  void init() { clear(); }

  /**
   *  \brief buffer for nelems elements of T, aligned to kAlign
   **/
  T *alloc(std::size_t nelems) {
    const std::size_t bytes = nelems * sizeof(T);
    if (bytes > kMaxClassBytes) return direct_alloc(bytes);
    const int c = class_of(bytes);
    ThreadCache &tc = cache();
    if (tc.head[c]) {
      T *p = tc.head[c];
      tc.head[c] = next_of(p);
      --tc.count[c];
      return checked(p);
    }
    // bulk refill from the central freelist: one lock per kBulk blocks
    {
      std::lock_guard<std::mutex> guard(mtx_);
      std::vector<T *> &cf = central_[c];
      if (!cf.empty()) {
        std::size_t take = cf.size() < kBulk ? cf.size() : kBulk;
        T *p = cf.back();
        cf.pop_back();
        for (--take; take > 0; --take) {
          set_next(cf.back(), tc.head[c]);
          tc.head[c] = cf.back();
          cf.pop_back();
          ++tc.count[c];
        }
        return checked(p);
      }
    }
    // bump-carve a fresh block from the thread's current chunk
    const std::size_t sz = kClassSizes[c];
    if (std::size_t(tc.bump_end - tc.bump) < sz) new_chunk(tc);
    T *p = reinterpret_cast<T *>(tc.bump);
    tc.bump += sz;
    return checked(p);
  }

  /**
   *  \brief return a buffer previously obtained from alloc(nelems)
   **/
  void free(T *p, std::size_t nelems) {
    const std::size_t bytes = nelems * sizeof(T);
    if (bytes > kMaxClassBytes) return direct_free(p, bytes);
    const int c = class_of(bytes);
    ThreadCache &tc = cache();
    set_next(p, tc.head[c]);
    tc.head[c] = p;
    if (++tc.count[c] >= kLocalMax) spill(tc, c);
  }

  /**
   *  \brief bytes currently held by the pool (nothing is returned to the
   *         OS before clear(), so this is also the high-water mark)
   **/
  std::size_t footprint() const {
    std::lock_guard<std::mutex> guard(mtx_);
    return chunks_.size() * kChunkBytes + direct_bytes_;
  }

  void clear() {
    // invalidate all thread_local registrations first: a stale entry can
    // then never match and is re-registered lazily
    epoch_.store(next_epoch(), std::memory_order_release);
    caches_.clear();
    for (void *m : chunks_) ::operator delete(m, std::align_val_t(kAlign));
    for (void *m : direct_) ::operator delete(m, std::align_val_t(kAlign));
    chunks_.clear();
    direct_.clear();
    direct_bytes_ = 0;
    for (auto &cf : central_) cf.clear();
  }

 private:
  struct alignas(kAlign) ThreadCache {
    T *head[kNumClasses] = {};
    std::uint32_t count[kNumClasses] = {};
    char *bump = nullptr;
    char *bump_end = nullptr;
  };

  static std::uint64_t next_epoch() {
    static std::atomic<std::uint64_t> ctr{1};
    return ctr.fetch_add(1);
  }

  // resolve the calling thread's cache; registered lazily once per
  // thread and pool generation, so the hot path is a thread_local read
  ThreadCache &cache() {
    struct TlsEntry {
      const void *owner;
      std::uint64_t epoch;
      ThreadCache *cache;
    };
    thread_local std::vector<TlsEntry> tls;
    const std::uint64_t ep = epoch_.load(std::memory_order_acquire);
    for (auto &e : tls)
      if (e.owner == this) {
        if (e.epoch == ep) return *e.cache;
        e.cache = register_cache();
        e.epoch = ep;
        return *e.cache;
      }
    ThreadCache *c = register_cache();
    tls.push_back({this, ep, c});
    return *c;
  }

  ThreadCache *register_cache() {
    std::unique_ptr<ThreadCache> c(new ThreadCache());
    ThreadCache *p = c.get();
    std::lock_guard<std::mutex> guard(mtx_);
    caches_.push_back(std::move(c));
    return p;
  }

  static T *checked(T *p) {
    assert((reinterpret_cast<std::uintptr_t>(p) & (kAlign - 1)) == 0 &&
           "MemoryPool returned a misaligned block");
    return p;
  }

  // the next pointer of a free block lives in its first bytes
  static T *next_of(const T *p) {
    T *n;
    std::memcpy(&n, p, sizeof(n));
    return n;
  }
  static void set_next(T *p, T *n) { std::memcpy(p, &n, sizeof(n)); }

  // move half of an overfull local freelist to central with one lock
  void spill(ThreadCache &tc, int c) {
    T *batch[kLocalMax / 2];
    for (std::size_t i = 0; i < kLocalMax / 2; ++i) {
      batch[i] = tc.head[c];
      tc.head[c] = next_of(batch[i]);
    }
    tc.count[c] -= kLocalMax / 2;
    std::lock_guard<std::mutex> guard(mtx_);
    central_[c].insert(central_[c].end(), batch, batch + kLocalMax / 2);
  }

  // chunks are independent allocations and are never moved or freed
  // before clear(); the remainder of the previous chunk is abandoned
  void new_chunk(ThreadCache &tc) {
    char *m = static_cast<char *>(
        ::operator new(kChunkBytes, std::align_val_t(kAlign)));
    tc.bump = m;
    tc.bump_end = m + kChunkBytes;
    std::lock_guard<std::mutex> guard(mtx_);
    chunks_.push_back(m);
  }

  // blocks above kMaxClassBytes bypass the classes (rare by design)
  T *direct_alloc(std::size_t bytes) {
    const std::size_t rounded = (bytes + kAlign - 1) / kAlign * kAlign;
    void *p = ::operator new(rounded, std::align_val_t(kAlign));
    std::lock_guard<std::mutex> guard(mtx_);
    direct_.push_back(p);
    direct_bytes_ += rounded;
    return checked(static_cast<T *>(p));
  }
  void direct_free(T *p, std::size_t bytes) {
    const std::size_t rounded = (bytes + kAlign - 1) / kAlign * kAlign;
    {
      std::lock_guard<std::mutex> guard(mtx_);
      for (std::size_t i = 0; i < direct_.size(); ++i)
        if (direct_[i] == p) {
          direct_[i] = direct_.back();
          direct_.pop_back();
          direct_bytes_ -= rounded;
          break;
        }
    }
    ::operator delete(p, std::align_val_t(kAlign));
  }

  mutable std::mutex mtx_;
  std::atomic<std::uint64_t> epoch_;
  std::vector<std::unique_ptr<ThreadCache>> caches_;
  std::array<std::vector<T *>, kNumClasses> central_;
  std::vector<void *> chunks_;
  std::vector<void *> direct_;
  std::size_t direct_bytes_ = 0;
};

}  // namespace FMCA
#endif

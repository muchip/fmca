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
// Simplified drop-in replacement for MemoryPool: no size classes, no
// per-thread local caches, no bump allocator. Every acquire()/release() call
// simply forwards to Eigen's aligned malloc/free.
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

  MemoryPool() {}

  // nthreads and max_alloc_mb are accepted for interface compatibility but
  // are unused: this dummy pool tracks no per-thread state and imposes no
  // chunking/size limits, it just allocates directly on demand.
  MemoryPool(Index max_elems, Index nthreads, Index max_alloc_mb = 0) {
    init(max_elems, nthreads, max_alloc_mb);
  }

  void init(Index /*max_elems*/, Index /*nthreads*/,
            Index /*max_alloc_mb*/ = 0) {
    // nothing to set up: every request is served by a fresh aligned_malloc
  }

  T *acquire(Index elems, Index /*tid*/ = 0) {
    return static_cast<T *>(Eigen::internal::aligned_malloc(elems * sizeof(T)));
  }

  void release(T *p, Index /*elems*/ = 0, Index /*tid*/ = 0) {
    Eigen::internal::aligned_free(p);
  }

  void clear() {
    // no persistent state to release: acquire()/release() are already
    // one-shot Eigen aligned_malloc/aligned_free calls
  }
};

}  // namespace FMCA
#endif

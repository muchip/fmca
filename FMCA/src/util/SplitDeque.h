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
#ifndef FMCA_UTIL_SPLITDEQUE_H_
#define FMCA_UTIL_SPLITDEQUE_H_

#include "Macros.h"

namespace FMCA {

/**
 *  \brief split deque for work stealing, cf. Dinan et al. SC'09 and
 *         van Dijk/van de Pol, Euro-Par 2014.
 *
 *  A single ring buffer with a movable split point:
 *
 *      [head_, split_)   shared, thieves take from head_ (oldest, coarsest)
 *      [split_, tail_)   private, owner pushes and pops at tail_ (LIFO)
 *
 *  head_, split_ and tail_ live in a logical index space and never wrap; only
 *  the array access wraps, via & mask_. Hence tail_ - head_ is always the true
 *  occupancy and full is distinguishable from empty without a spare slot.
 *
 *  The invariant head_ <= split_ <= tail_ makes the two regions disjoint
 *  intervals, so no slot is nameable by both the owner and a thief. That is
 *  what allows push and pop to touch no atomic read-modify-write and no fence:
 *  push writes a slot and bumps a plain index, pop compares tail_ against one
 *  relaxed load of split_. Only publish (split_ up), reclaim (split_ down) and
 *  steal synchronise, and batching makes those rare.
 *
 *  Moving split_ up is lock free: it only reveals slots that were already
 *  written, so a release store paired with the acquire in steal suffices.
 *  Moving it down reclaims slots a thief may be about to read, which ordering
 *  alone cannot make safe, hence the mutex.
 *
 *  LIFO at the private end is what bounds memory: the parent just made ready
 *  is popped next, so a worker climbs one branch and the sons' payloads are
 *  released immediately. Everything above split_ is a region a thief can open,
 *  so kHighWater is a direct knob on peak memory.
 **/
template <typename T>
class SplitDeque {
 public:
  SplitDeque(const SplitDeque &) = delete;
  SplitDeque &operator=(const SplitDeque &) = delete;

  //  private entries beyond which the owner publishes
  static constexpr std::ptrdiff_t kHighWater = 4;

  SplitDeque() { init(1024); }
  explicit SplitDeque(std::ptrdiff_t capacity) { init(capacity); }

  /**
   *  \brief size the ring buffer to the next power of two >= capacity.
   **/
  void init(std::ptrdiff_t capacity) {
    std::ptrdiff_t c = 1;
    while (c < capacity) c <<= 1;
    a_.assign(c, nullptr);
    mask_ = c - 1;
    high_water_ = std::max<std::ptrdiff_t>(1, std::min(+kHighWater, c / 2));
    tail_ = 0;
    head_.store(0, std::memory_order_relaxed);
    split_.store(0, std::memory_order_relaxed);
  }

  /**
   *  \brief push onto the private end. No synchronisation.
   *
   *  A stale head_ is always stale low, since thieves only increase it, so the
   *  capacity check errs towards reporting full and tail_ & mask_ can never
   *  name a slot a thief still holds.
   **/
  bool push(T *v) {
    if (tail_ - head_.load(std::memory_order_acquire) > mask_) return false;
    a_[tail_ & mask_] = v;
    ++tail_;
    //  Two triggers. The first bounds the private region. The second matters
    //  more: under LIFO the private depth hovers around the tree depth and
    //  never approaches high_water_, so without it split_ would stay pinned
    //  where the seeds left it, nothing new would ever become stealable and
    //  work stealing would degrade to draining the initial seeds.
    const std::ptrdiff_t priv = tail_ - split_.load(std::memory_order_relaxed);
    if (priv > high_water_ || (priv > 1 && shared_empty())) publish();
    return true;
  }

  /**
   *  \brief pop the newest private entry, else reclaim a batch from the shared
   *         portion and pop from that. Returns nullptr if the deque is empty.
   **/
  T *pop() {
    if (tail_ > split_.load(std::memory_order_relaxed))
      return a_[--tail_ & mask_];
    if (!reclaim()) return nullptr;
    //  reaching here means tail_ == split_, and a successful reclaim strictly
    //  lowers split_, so the private region is non-empty
    return a_[--tail_ & mask_];
  }

  /**
   *  \brief hand the oldest half of the private portion to the thieves. Only
   *         the owner writes split_ and the slots are already committed, so a
   *         release store suffices and no lock is taken.
   **/
  void publish() {
    const std::ptrdiff_t s = split_.load(std::memory_order_relaxed);
    const std::ptrdiff_t n = tail_ - s;
    if (n < 2) return;
    split_.store(s + n / 2, std::memory_order_release);
  }

  /**
   *  \brief take the newest half of the shared portion back. Moves split_
   *         down, which can race a thief ->lock.
   **/
  bool reclaim() {
    if (shared_empty()) return false;
    std::lock_guard<std::mutex> lock(m_);
    const std::ptrdiff_t h = head_.load(std::memory_order_relaxed);
    const std::ptrdiff_t s = split_.load(std::memory_order_relaxed);
    if (h >= s) return false;
    split_.store(s - (s - h + 1) / 2, std::memory_order_release);
    return true;
  }

  /**
   *  \brief take the oldest shared entry, i.e. the work furthest from the
   *         owner's active branch.
   **/
  T *steal() {
    if (shared_empty()) return nullptr;
    std::lock_guard<std::mutex> lock(m_);
    const std::ptrdiff_t h = head_.load(std::memory_order_relaxed);
    if (h >= split_.load(std::memory_order_acquire)) return nullptr;
    T *v = a_[h & mask_];
    head_.store(h + 1, std::memory_order_release);
    return v;
  }

  /**
   *  \brief unlocked hint, may be stale in either direction. For the owner it
   *         can only err towards true, since thieves only raise head_.
   **/
  bool shared_empty() const {
    return head_.load(std::memory_order_acquire) >=
           split_.load(std::memory_order_acquire);
  }

 private:
  std::vector<T *> a_;
  std::ptrdiff_t mask_ = 0;
  std::ptrdiff_t high_water_ = 0;
  //  128 rather than 64: x86 L2 prefetches adjacent line pairs, and Apple
  //  silicon and POWER use 128 byte lines
  alignas(128) std::ptrdiff_t tail_ = 0;
  alignas(128) std::atomic<std::ptrdiff_t> head_{0};
  alignas(128) std::atomic<std::ptrdiff_t> split_{0};
  alignas(128) mutable std::mutex m_;
};

}  // namespace FMCA
#endif

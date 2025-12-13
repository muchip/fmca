// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_CLUSTERING_GREEDYSETCOVERING_H_
#define FMCA_CLUSTERING_GREEDYSETCOVERING_H_

namespace FMCA {

class PriorityQueue {
 public:
  explicit PriorityQueue(Index n) : pos_(n, -1) { heap_.reserve(n); }
  bool empty() const { return heap_.empty(); }

  void push(Index key, Index ind) {
    heap_.push_back(std::make_pair(key, ind));
    const Index k = heap_.size() - 1;
    pos_[ind] = k;
    siftUp(k);
  }

  std::pair<Index, Index> top() const { return heap_.front(); }

  void pop() {
    const Index last = heap_.size() - 1;
    swapNodes(0, last);
    pos_[heap_[last].second] = -1;
    heap_.pop_back();
    if (not heap_.empty()) siftDown(0);
  }

  void decreaseKey(Index i) {
    const Index k = pos_[i];
    if (k == -1 || heap_[k].first == 0) return;
    --(heap_[k].first);
    siftDown(k);
  }

 private:
  void siftUp(Index k) {
    while (k > 0) {
      const Index p = (k - 1) / 2;
      if (!(heap_[p].first < heap_[k].first)) break;
      swapNodes(p, k);
      k = p;
    }
  }

  void siftDown(Index k) {
    while (true) {
      const Index l = 2 * k + 1;
      const Index r = 2 * k + 2;
      Index best = k;
      if (l < heap_.size() && heap_[best] < heap_[l]) best = l;
      if (r < heap_.size() && heap_[best] < heap_[r]) best = r;
      if (best == k) break;
      swapNodes(k, best);
      k = best;
    }
  }

  void swapNodes(Index a, Index b) {
    std::swap(heap_[a], heap_[b]);
    pos_[heap_[a].second] = a;
    pos_[heap_[b].second] = b;
  }

  std::vector<std::pair<Index, Index>> heap_;
  std::vector<Index> pos_;
};

/**
 *  \ingroup Clustering
 *  \brief uses the cluster tree ct to efficiently determine a
 *         set covering of a given radius
 **/
template <typename Derived>
std::vector<Index> greedySetCovering(const ClusterTreeBase<Derived> &ct,
                                     const Matrix &P, const Scalar r) {
  std::vector<Index> retval;
  std::vector<bool> is_covered(P.cols(), false);
  std::vector<std::vector<Index>> rballs(P.cols());
  std::vector<std::vector<Index>> index_covers(P.cols());
  Index num_covered = 0;

#pragma omp parallel for
  for (Index i = 0; i < rballs.size(); ++i) {
    rballs[i] = epsNN(ct, P, P.col(i), 0.5 * r);
    for (const auto &it : rballs[i])
#pragma omp critical
      index_covers[it].push_back(i);
  }
  PriorityQueue heap(P.cols());
  // create a max-heap to order points by n_uncovered
  for (Index i = 0; i < P.cols(); ++i)
    if (rballs[i].size() > 0) heap.push(rballs[i].size(), i);

  while (num_covered != P.cols()) {
    std::pair<Index, Index> top = heap.top();
    heap.pop();
    num_covered += top.first;
    retval.push_back(top.second);
    for (const auto &it : rballs[top.second]) {
      // reduce uncovered size of affected balls
      if (!is_covered[it]) {
        for (const auto &it2 : index_covers[it]) heap.decreaseKey(it2);
        // mask covered indices
        is_covered[it] = true;
      }
    }
  }
  return retval;
}

}  // namespace FMCA
#endif

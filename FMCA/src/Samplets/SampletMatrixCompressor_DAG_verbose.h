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
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSORDAG_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSORDAG_H_

#include "../util/CompressorDAG.h"
#include "recursivelyComputeBlock.h"

namespace FMCA {
/**
 *  \brief samplet compressor on the CompressorDAG with a column-subtree
 *         schedule, symmetric and unsymmetric
 **/
template <typename H2STreeType, typename ClusterComparison = CompareCluster>
class SampletMatrixCompressor
    : public SampletMatrixCompressorBase<
          SampletMatrixCompressor<H2STreeType, ClusterComparison>> {
 public:
  typedef SampletMatrixCompressorBase<
      SampletMatrixCompressor<H2STreeType, ClusterComparison>>
      Base;
  using Base::aposteriori_triplets;
  using Base::aposteriori_triplets_fast;
  using Base::cols;
  using Base::rows;
  using Base::triplets;

  typedef internal::CompressorDAG<H2STreeType, ClusterComparison> DAG;
  typedef typename DAG::Node Node;
  typedef typename DAG::PatternIdx PatternIdx;
  typedef typename DAG::Strategy Strategy;

  SampletMatrixCompressor() {}
  SampletMatrixCompressor(const SampletTreeBase<H2STreeType> &ST, Scalar eta,
                          Scalar threshold = 0) {
    init(ST, eta, threshold);
  }
  SampletMatrixCompressor(const SampletTreeBase<H2STreeType> &TR,
                          const SampletTreeBase<H2STreeType> &TC, Scalar eta,
                          Scalar threshold = 0, bool sym = false) {
    init(TR, TC, eta, threshold, sym);
  }

  const DAG &dag() const { return dag_; }

  // symmetric, one tree, upper block triangle
  void init(const SampletTreeBase<H2STreeType> &ST, Scalar eta,
            Scalar threshold = 0) {
    init(ST, ST, eta, threshold, true);
    return;
  }

  // unsymmetric, two trees, full matrix
  void init(const SampletTreeBase<H2STreeType> &TR,
            const SampletTreeBase<H2STreeType> &TC, Scalar eta,
            Scalar threshold = 0, bool sym = false) {
    std::cout << "using compressor 4" << std::endl;
    Base::setDimensions(TR.block_size(), TC.block_size());
    Base::setThreshold(threshold);
    Base::setEta(eta);
    dag_.init(TR.derived(), TC.derived(), eta, sym);
    return;
  }

  /**
   *  \brief compressor function using omp parallelism
   *
   *         cut_level < 0 picks the first level with at least
   *         4 * nthreads columns and performs parallelization over these
   *         subtrees
   **/
  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen, std::ptrdiff_t cut_level = -1) {
    assert(dag_.nodes().size() && "compress: DAG empty, call init first");
    const Index nthreads = omp_get_max_threads();
    const internal::RandomTreeAccessor<H2STreeType> &rta = dag_.c_rta();
    const Index L = rta.max_level();
    // rta levels()[l]..levels()[l+1] are the block ids of column level l
    const std::vector<Index> &lvls = rta.levels();
    Index l0 = 0;
    if (cut_level >= 0)
      l0 = std::min<Index>(cut_level, L);
    else
      while (l0 < L && lvls[l0 + 1] - lvls[l0] < 4 * nthreads) ++l0;

    std::vector<Worker> workers(nthreads);
    Base::clearTriplets();
    alive_.store(0, std::memory_order_relaxed);
    peak_alive_.store(0, std::memory_order_relaxed);

    // phase 1: one closed task per column subtree rooted at level l0
#pragma omp parallel for schedule(dynamic)
    for (PatternIdx c = lvls[l0]; c < PatternIdx(lvls[l0 + 1]); ++c)
      processSubtree(*rta.nodes()[c], e_gen, workers[omp_get_thread_num()]);
    // phase 2: the columns above the cut, one round per level. The sons of
    // a column on level l are done after round l+1 (or phase 1), and the
    // implicit barrier of the loop orders the rounds
    for (Index l = l0; l > 0; --l) {
#pragma omp parallel for schedule(dynamic)
      for (PatternIdx c = lvls[l - 1]; c < PatternIdx(lvls[l]); ++c)
        processColumn(c, e_gen, workers[omp_get_thread_num()]);
    }

    {
      Stats s;
      for (const Worker &w : workers) s += w.stats;
      const std::ptrdiff_t nnodes = dag_.nodes().size();
      const std::ptrdiff_t sons = s.recycled + s.lowrank + s.full;
      const double d = nnodes ? 100.0 / double(nnodes) : 0.0;
      const double q = sons ? 100.0 / double(sons) : 0.0;
      const std::ptrdiff_t peak = peak_alive_.load(std::memory_order_relaxed);
      std::cout << "nodes                         " << nnodes << "\n"
                << "cut level / tasks             " << l0 << " / "
                << lvls[l0 + 1] - lvls[l0] << " (max level " << L << ")\n"
                << "strategy rows / cols / leaf   " << s.rows << " ("
                << s.rows * d << "%) / " << s.cols << " (" << s.cols * d
                << "%) / " << s.leaf << " (" << s.leaf * d << "%)\n"
                << "sons recycled / lowrank / full " << s.recycled << " ("
                << s.recycled * q << "%) / " << s.lowrank << " ("
                << s.lowrank * q << "%) / " << s.full << " (" << s.full * q
                << "%)\n"
                << "peak blocks alive             " << peak << " (" << peak * d
                << "%)" << std::endl;
      assert(alive_.load(std::memory_order_relaxed) == 0 &&
             "compress: blocks left alive");
    }
    // computation done. delete the dag
    std::vector<Node>().swap(dag_.nodes());
    // merge the per thread buffers
    {
      std::size_t total = 0;
      for (const Worker &w : workers) total += w.triplets.size();
      Base::reserveTriplets(total);
      for (Worker &w : workers) {
        Base::appendTriplets(std::move(w.triplets));
        std::vector<Triplet>().swap(w.triplets);
      }
    }
    return;
  }

 private:
  struct Stats {
    std::ptrdiff_t rows = 0;
    std::ptrdiff_t cols = 0;
    std::ptrdiff_t leaf = 0;
    std::ptrdiff_t recycled = 0;  // son block read from the DAG
    std::ptrdiff_t lowrank = 0;   // missing son, one interpolation
    std::ptrdiff_t full = 0;      // missing son, full recursion (filtered
                                  // fringe, row orphans)
    Stats &operator+=(const Stats &o) {
      rows += o.rows;
      cols += o.cols;
      leaf += o.leaf;
      recycled += o.recycled;
      lowrank += o.lowrank;
      full += o.full;
      return *this;
    }
  };
  // per thread state, padded so that the hot counters of two threads never
  // share a cache line
  struct alignas(128) Worker {
    std::vector<Triplet> triplets;
    Stats stats;
  };

  template <typename EntGen>
  void processSubtree(const H2STreeType &c, const EntGen &e_gen, Worker &w) {
    for (Index i = 0; i < c.nSons(); ++i) processSubtree(c.sons(i), e_gen, w);
    processColumn(c.block_id(), e_gen, w);
    return;
  }

  // rows of column c in descending level: a Rows node reads deeper rows of
  // the same column, which then precede it
  template <typename EntGen>
  void processColumn(PatternIdx c, const EntGen &e_gen, Worker &w) {
    const PatternIdx *outer = dag_.pattern().outerIndexPtr();
    const PatternIdx *val = dag_.pattern().valuePtr();
    for (PatternIdx p = outer[c + 1]; p-- > outer[c];)
      computeNode(val[p], e_gen, w);
    return;
  }

  template <typename EntGenerator>
  void computeNode(PatternIdx k, const EntGenerator &e_gen, Worker &w) {
    std::vector<Node> &nodes = dag_.nodes();
    Node &v = nodes[k];
    const H2STreeType *pr = v.pr;
    const H2STreeType *pc = v.pc;
    assert(v.deps.load(std::memory_order_relaxed) == 0 &&
           "computeNode: son not finished, schedule broken");
    assert(!v.block.size() && "computeNode: node computed twice");
    switch (v.strategy) {
      case Strategy::Leaf: {
        ++w.stats.leaf;
        v.block = computeBlock(*pr, *pc, e_gen);
        break;
      }
      case Strategy::Rows: {
        ++w.stats.rows;
        Matrix buf(pr->Q().rows(), pc->Q().cols());
        Index offset = 0;
        for (Index i = 0; i < pr->nSons(); ++i) {
          const Index nscalfs = pr->sons(i).nscalfs();
          const PatternIdx s = v.sons[i];
          if (s >= 0) {
            ++w.stats.recycled;
            buf.middleRows(offset, nscalfs) = nodes[s].block.topRows(nscalfs);
          } else {
            countRecompute(pr->sons(i), *pc, w);
            buf.middleRows(offset, nscalfs) =
                computeBlock(pr->sons(i), *pc, e_gen).topRows(nscalfs);
          }
          offset += nscalfs;
        }
        v.block.noalias() = pr->Q().transpose() * buf;
        break;
      }
      case Strategy::Cols: {
        ++w.stats.cols;
        Matrix buf(pr->Q().cols(), pc->Q().rows());
        Index offset = 0;
        for (Index i = 0; i < pc->nSons(); ++i) {
          const Index nscalfs = pc->sons(i).nscalfs();
          const PatternIdx s = v.sons[i];
          if (s >= 0) {
            ++w.stats.recycled;
            buf.middleCols(offset, nscalfs) = nodes[s].block.leftCols(nscalfs);
          } else {
            countRecompute(*pr, pc->sons(i), w);
            buf.middleCols(offset, nscalfs) =
                computeBlock(*pr, pc->sons(i), e_gen).leftCols(nscalfs);
          }
          offset += nscalfs;
        }
        v.block.noalias() = buf * pc->Q();
        break;
      }
    }
    countAlloc();
    storeBlock(*pr, *pc, w.triplets, v.block);
    // first sons, then block itself. fetch_sub returns the undecremented value,
    // 1 identifies the last decrementer, which frees. Within a task all
    // of this is one thread, across tasks the only shared blocks are task
    // roots read after the barrier, so the atomics are uncontended
    for (const PatternIdx s : v.sons)
      if (s >= 0 &&
          nodes[s].consumers.fetch_sub(1, std::memory_order_acq_rel) == 1)
        releaseBlock(nodes[s].block);
    if (v.consumers.fetch_sub(1, std::memory_order_acq_rel) == 1)
      releaseBlock(v.block);
    // readiness is provided by the order, not by the counters. The dads'
    // deps are decremented anyway so that the assert above can check the
    // schedule; two uncontended atomics per node
    if (v.row_consumer)
      nodes[v.row_dad].deps.fetch_sub(1, std::memory_order_acq_rel);
    if (v.col_consumer)
      nodes[v.col_dad].deps.fetch_sub(1, std::memory_order_acq_rel);
    return;
  }

  template <typename EntGen>
  Matrix computeBlock(const H2STreeType &TR, const H2STreeType &TC,
                      const EntGen &e_gen) const {
    return internal::recursivelyComputeBlock<H2STreeType, EntGen,
                                             ClusterComparison>(TR, TC, e_gen,
                                                                Base::eta());
  }

  void countRecompute(const H2STreeType &TR, const H2STreeType &TC,
                      Worker &w) const {
    if (ClusterComparison::compare(TR, TC, Base::eta()) == LowRank)
      ++w.stats.lowrank;
    else
      ++w.stats.full;
  }

  inline void storeBlock(const H2STreeType &TR, const H2STreeType &TC,
                         std::vector<Triplet> &triplet_buffer,
                         const Matrix &block) {
    const Index nrows = TR.is_root() ? TR.Q().cols() : TR.nsamplets();
    const Index ncols = TC.is_root() ? TC.Q().cols() : TC.nsamplets();
    if (dag_.sym())
      Base::storeSymTriplets(triplet_buffer, TR.start_index(), TC.start_index(),
                             nrows, ncols,
                             block.bottomRightCorner(nrows, ncols));
    else
      Base::storeTriplets(triplet_buffer, TR.start_index(), TC.start_index(),
                          nrows, ncols, block.bottomRightCorner(nrows, ncols));
  }

  void releaseBlock(Matrix &block) {
    Matrix().swap(block);
    alive_.fetch_sub(1, std::memory_order_relaxed);
  }
  void countAlloc() {
    const std::ptrdiff_t a = alive_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::ptrdiff_t p = peak_alive_.load(std::memory_order_relaxed);
    while (p < a && !peak_alive_.compare_exchange_weak(
                        p, a, std::memory_order_relaxed)) {
    }
  }

  DAG dag_;
  std::atomic<std::ptrdiff_t> alive_{0};
  std::atomic<std::ptrdiff_t> peak_alive_{0};
};
}  // namespace FMCA

#endif

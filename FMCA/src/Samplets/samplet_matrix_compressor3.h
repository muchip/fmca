// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSOR_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSOR_H_

#include "../util/CompressorDAG.h"
#include "../util/MemoryPool2.h"
#include "../util/SplitDeque.h"

namespace FMCA {
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

  typedef AMap<Matrix> MMatrix;
  typedef internal::CompressorDAG<H2STreeType, MMatrix, ClusterComparison> DAG;
  SampletMatrixCompressor() {}
  SampletMatrixCompressor(const SampletTreeBase<H2STreeType> &ST, Scalar eta,
                          Scalar threshold = 0) {
    init(ST, eta, threshold);
  }

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   *
   **/
  void init(const SampletTreeBase<H2STreeType> &ST, Scalar eta,
            Scalar threshold = 0) {
    typedef typename DAG::Node Node;
    std::cout << "using compressor 3" << std::endl;
    Base::setDimensions(ST.block_size(), ST.block_size());
    Base::setThreshold(threshold);
    Base::setEta(eta);
    dag_.init(ST.derived(), ST.derived(), eta, true);
    max_size_ = 0;

    // sweep to fix strategy and maximum memory size
    sources_.clear();
    for (Node &v : dag_.nodes()) {
      max_size_ = std::max<std::ptrdiff_t>(
          {max_size_, v.pr->Q().rows(), v.pr->Q().cols(), v.pr->V().rows(),
           v.pc->Q().rows(), v.pc->Q().cols(), v.pc->V().rows()});
      Index nrow = 0;
      Index ncol = 0;
      for (const Node *s : v.row_sons) nrow += (s != nullptr);
      for (const Node *s : v.col_sons) ncol += (s != nullptr);
#if 0
      if (nrow == 0 && ncol == 0)
        v.strategy = Node::Leaf;
      else
        v.strategy = (ncol >= nrow) ? Node::Cols : Node::Rows;
#endif
      if (ncol)
        v.strategy = Node::Cols;
      else if (nrow)
        v.strategy = Node::Rows;
      else
        v.strategy = Node::Leaf;
      const Index deps = (v.strategy == Node::Rows ? nrow : ncol);
      v.deps_remaining.store(deps, std::memory_order_relaxed);
      v.consumers_remaining.store(0, std::memory_order_relaxed);
      if (!deps) sources_.push_back(std::addressof(v));
    }

    // dependent on the strategy of the parent fix consumer count of children
    for (Node &v : dag_.nodes()) {
      if (v.strategy == Node::Leaf) continue;
      for (Node *s : (v.strategy == Node::Rows ? v.row_sons : v.col_sons))
        if (s != nullptr)
          s->consumers_remaining.fetch_add(1, std::memory_order_relaxed);
    }

    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
    typedef typename DAG::Node Node;
    assert(dag_.nodes().size() && "compress: DAG empty, call init first");
    const Index nthreads = omp_get_max_threads();
    const std::ptrdiff_t nnodes = dag_.nodes().size();
    const std::ptrdiff_t nsources = sources_.size();
    std::vector<std::vector<Triplet>> tlist(nthreads);
    //  SplitDeque has no copy or move, so the vector is sized once here and
    //  never resized. Capacity only has to hold the readied frontier of one
    //  worker: the sources live in sources_ and are handed out below
    std::vector<SplitDeque<Node>> queues(nthreads);
    mem_arena_.init(max_size_ * max_size_, nthreads);
    Base::clearTriplets();
    std::atomic<std::ptrdiff_t> remaining(nnodes);
    std::atomic<std::ptrdiff_t> next_source(0);
    //  diagnostics, thread local and folded in once at the end of the region
    std::atomic<std::ptrdiff_t> n_spin(0);
    std::atomic<std::ptrdiff_t> n_from_pop(0);
    std::atomic<std::ptrdiff_t> n_from_steal(0);
    std::atomic<std::ptrdiff_t> n_tail(0);
    std::atomic<std::ptrdiff_t> n_leaf(0);
    std::atomic<std::ptrdiff_t> n_rows(0);
    std::atomic<std::ptrdiff_t> n_cols(0);
    std::atomic<std::ptrdiff_t> n_recycled(0);
    std::atomic<std::ptrdiff_t> n_recomputed(0);

#pragma omp parallel num_threads(nthreads)
    {
      const Index tid = omp_get_thread_num();
      SplitDeque<Node> &myq = queues[tid];
      std::ptrdiff_t l_spin = 0;
      std::ptrdiff_t l_pop = 0;
      std::ptrdiff_t l_steal = 0;
      std::ptrdiff_t l_tail = 0;
      std::ptrdiff_t l_leaf = 0;
      std::ptrdiff_t l_rows = 0;
      std::ptrdiff_t l_cols = 0;
      std::ptrdiff_t l_recycled = 0;
      std::ptrdiff_t l_recomputed = 0;

      while (remaining.load(std::memory_order_acquire) > 0) {
        //  own readied work first: pop takes the newest, which is the parent
        //  just made ready by the node we finished, so the worker climbs one
        //  branch and the sons' payloads are released as it goes
        Node *v = myq.pop();
        if (v != nullptr) ++l_pop;
        //  then a source. One fetch_add hands each index to exactly one
        //  thread, so there is no static partition to get unlucky with and no
        //  rescan of the node array
        if (v == nullptr &&
            next_source.load(std::memory_order_relaxed) < nsources) {
          const std::ptrdiff_t i =
              next_source.fetch_add(1, std::memory_order_relaxed);
          if (i < nsources) v = sources_[i];
        }
        //  then somebody else's oldest shared work
        if (v == nullptr) {
          for (Index k = 1; v == nullptr && k < nthreads; ++k)
            v = queues[(tid + k) % nthreads].steal();
          if (v != nullptr) ++l_steal;
        }
        if (v == nullptr) {
          ++l_spin;
          continue;
        }
        //  work done after the source reservoir is exhausted. If this is small
        //  the DAG genuinely runs dry and the tail is its critical path, so no
        //  scheduling change helps and the spin is pure waste. If it is large
        //  there is parallel work the workers are failing to find
        if (next_source.load(std::memory_order_relaxed) >= nsources) ++l_tail;

        const H2STreeType *pr = v->pr;
        const H2STreeType *pc = v->pc;
        new (&v->block)
            MMatrix(acquireMap(pr->Q().cols(), pc->Q().cols(), tid));

        switch (v->strategy) {
          case Node::Leaf:
            ++l_leaf;
            recursivelyComputeBlock_noalloc(*pr, *pc, e_gen, v->block, tid);
            break;
          case Node::Rows: {
            ++l_rows;
            MMatrix buf = acquireMap(pr->Q().rows(), pc->Q().cols(), tid);
            Index offset = 0;
            for (Index k = 0; k < pr->nSons(); ++k) {
              const Index nscalfs = pr->sons(k).nscalfs();
              if (v->row_sons[k] != nullptr) {
                ++l_recycled;
                buf.middleRows(offset, nscalfs) =
                    v->row_sons[k]->block.topRows(nscalfs);
              } else {
                ++l_recomputed;
                MMatrix temp =
                    acquireMap(pr->sons(k).Q().cols(), pc->Q().cols(), tid);
                recursivelyComputeBlock_noalloc(pr->sons(k), *pc, e_gen, temp,
                                                tid);
                buf.middleRows(offset, nscalfs) = temp.topRows(nscalfs);
                releaseMap(temp, tid);
              }
              offset += nscalfs;
            }
            v->block.noalias() = pr->Q().transpose() * buf;
            releaseMap(buf, tid);
            break;
          }
          case Node::Cols: {
            ++l_cols;
            MMatrix buf = acquireMap(pr->Q().cols(), pc->Q().rows(), tid);
            Index offset = 0;
            for (Index k = 0; k < pc->nSons(); ++k) {
              const Index nscalfs = pc->sons(k).nscalfs();
              if (v->col_sons[k] != nullptr) {
                ++l_recycled;
                buf.middleCols(offset, nscalfs) =
                    v->col_sons[k]->block.leftCols(nscalfs);
              } else {
                ++l_recomputed;
                MMatrix temp =
                    acquireMap(pr->Q().cols(), pc->sons(k).Q().cols(), tid);
                recursivelyComputeBlock_noalloc(*pr, pc->sons(k), e_gen, temp,
                                                tid);
                buf.middleCols(offset, nscalfs) = temp.leftCols(nscalfs);
                releaseMap(temp, tid);
              }
              offset += nscalfs;
            }
            v->block.noalias() = buf * pc->Q();
            releaseMap(buf, tid);
            break;
          }
        }

        //  the node's own contribution. Only reads the block, so it never had
        //  to wait for the consumers and no longer sits in a gc sweep
        if (!pr->is_root() && !pc->is_root())
          storeSymBlock(
              tlist[tid], pr->start_index(), pc->start_index(), pr->nsamplets(),
              pc->nsamplets(),
              v->block.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
        else if (!pc->is_root())
          storeSymBlock(tlist[tid], pr->start_index(), pc->start_index(),
                        pr->Q().cols(), pc->nsamplets(),
                        v->block.rightCols(pc->nsamplets()));
        else
          storeSymBlock(tlist[tid], pr->start_index(), pc->start_index(),
                        pr->Q().cols(), pc->Q().cols(), v->block);

        //  fetch_sub returns the value before, so == 1 means we took it to
        //  zero and exactly one thread sees it
        for (Node *s : (v->strategy == Node::Rows ? v->row_sons : v->col_sons))
          if (s != nullptr && s->consumers_remaining.fetch_sub(
                                  1, std::memory_order_acq_rel) == 1)
            releaseMap(s->block, tid);
        //  nobody will ever decrement this one, so free it here or never
        if (!v->consumers_remaining.load(std::memory_order_acquire))
          releaseMap(v->block, tid);

        //  a parent only reads us if it chose our side. Pushed to our own
        //  deque, so pop takes it next and the climb continues
        if (v->row_dad != nullptr && v->row_dad->strategy == Node::Rows &&
            v->row_dad->deps_remaining.fetch_sub(
                1, std::memory_order_acq_rel) == 1)
          if (!myq.push(v->row_dad))
            assert(false && "compress: deque overflow, raise capacity");
        if (v->col_dad != nullptr && v->col_dad->strategy == Node::Cols &&
            v->col_dad->deps_remaining.fetch_sub(
                1, std::memory_order_acq_rel) == 1)
          if (!myq.push(v->col_dad))
            assert(false && "compress: deque overflow, raise capacity");

        remaining.fetch_sub(1, std::memory_order_release);
      }
      n_spin.fetch_add(l_spin, std::memory_order_relaxed);
      n_from_pop.fetch_add(l_pop, std::memory_order_relaxed);
      n_from_steal.fetch_add(l_steal, std::memory_order_relaxed);
      n_tail.fetch_add(l_tail, std::memory_order_relaxed);
      n_leaf.fetch_add(l_leaf, std::memory_order_relaxed);
      n_rows.fetch_add(l_rows, std::memory_order_relaxed);
      n_cols.fetch_add(l_cols, std::memory_order_relaxed);
      n_recycled.fetch_add(l_recycled, std::memory_order_relaxed);
      n_recomputed.fetch_add(l_recomputed, std::memory_order_relaxed);
    }
    {
      const std::ptrdiff_t pop = n_from_pop.load();
      const std::ptrdiff_t steal = n_from_steal.load();
      const std::ptrdiff_t rec = n_recycled.load();
      const std::ptrdiff_t rcp = n_recomputed.load();
      const double d = nnodes ? 100.0 / double(nnodes) : 0.0;
      const double s = (rec + rcp) ? 100.0 / double(rec + rcp) : 0.0;
      std::cout << "nodes                         " << nnodes << "\n"
                << "sources                       " << nsources << " ("
                << nsources * d << "%)\n"
                << "  from pop / source / steal   " << pop << " / "
                << nnodes - pop - steal << " / " << steal << "\n"
                << "  after sources exhausted     " << n_tail.load() << " ("
                << n_tail.load() * d << "%)\n"
                << "strategy rows / cols / leaf   " << n_rows.load() << " / "
                << n_cols.load() << " / " << n_leaf.load() << "\n"
                << "sons recycled / recomputed    " << rec << " (" << rec * s
                << "%) / " << rcp << " (" << rcp * s << "%)\n"
                << "idle spins                    " << n_spin.load() << " ("
                << double(n_spin.load()) / double(nnodes ? nnodes : 1)
                << " per node)" << std::endl;
    }
    dag_.nodes().clear();
    sources_.clear();
    for (Index i = 0; i < tlist.size(); ++i)
      Base::appendTriplets(std::move(tlist[i]));
    return;
  }

 private:
  using Base::storeSymBlock;

  /**
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  void recursivelyComputeBlock_noalloc(const H2STreeType &TR,
                                       const H2STreeType &TC,
                                       const EntryGenerator &e_gen,
                                       MMatrix &block, Index tid = 0) {
    // check for admissibility
    if (ClusterComparison::compare(TR, TC, Base::eta()) == LowRank) {
      MMatrix temp1 = acquireMap(TR.V().rows(), TC.V().rows(), tid);
      MMatrix temp2 = acquireMap(TR.V().rows(), TC.V().rows(), tid);
      MMatrix temp3 = acquireMap(TR.V().rows(), TC.V().cols(), tid);
      e_gen.interpolate_kernel_noalloc(TR, TC, temp1.data(), temp2.data());
      temp3.noalias() = temp1 * TC.V();
      block.noalias() = TR.V().transpose() * temp3;
      releaseMap(temp1, tid);
      releaseMap(temp2, tid);
      releaseMap(temp3, tid);
      return;
    } else {
      const Index the_case = 2 * (!TR.nSons()) + !TC.nSons();
      switch (the_case) {
        case 3: {
          // both are leafs: compute the block and return
          MMatrix temp1 = acquireMap(TR.Q().rows(), TC.Q().rows(), tid);
          MMatrix temp2 = acquireMap(TR.Q().rows(), TC.Q().cols(), tid);
          e_gen.compute_dense_block_noalloc(TR, TC, temp1.data());
          temp2.noalias() = temp1 * TC.Q();
          block.noalias() = TR.Q().transpose() * temp2;
          releaseMap(temp1, tid);
          releaseMap(temp2, tid);
          return;
        }
        case 2: {
          // the row cluster is a leaf cluster: recursion on the col cluster
          MMatrix temp1 = acquireMap(TR.Q().cols(), TC.Q().rows(), tid);

          Index offset = 0;
          for (Index j = 0; j < TC.nSons(); ++j) {
            MMatrix temp2 =
                acquireMap(TR.Q().cols(), TC.sons(j).Q().cols(), tid);
            recursivelyComputeBlock_noalloc(TR, TC.sons(j), e_gen, temp2, tid);
            const Index nscalfs = TC.sons(j).nscalfs();
            temp1.middleCols(offset, nscalfs) = temp2.leftCols(nscalfs);
            offset += nscalfs;
            releaseMap(temp2, tid);
          }
          block.noalias() = temp1 * TC.Q();
          releaseMap(temp1, tid);

          return;
        }
        case 1: {
          // the col cluster is a leaf cluster: recursion on the row cluster
          MMatrix temp1 = acquireMap(TR.Q().rows(), TC.Q().cols(), tid);
          Index offset = 0;
          for (Index i = 0; i < TR.nSons(); ++i) {
            MMatrix temp2 =
                acquireMap(TR.sons(i).Q().cols(), TC.Q().cols(), tid);
            recursivelyComputeBlock_noalloc(TR.sons(i), TC, e_gen, temp2, tid);
            const Index nscalfs = TR.sons(i).nscalfs();
            temp1.middleRows(offset, nscalfs) = temp2.topRows(nscalfs);
            offset += nscalfs;
            releaseMap(temp2, tid);
          }
          block.noalias() = TR.Q().transpose() * temp1;
          releaseMap(temp1, tid);
          return;
        }
        case 0: {
          // neither is a leaf, let recursion handle this
          MMatrix temp1 = acquireMap(TR.Q().rows(), TC.Q().cols(), tid);
          Index r_offset = 0;
          for (auto i = 0; i < TR.nSons(); ++i) {
            MMatrix temp2 =
                acquireMap(TR.sons(i).Q().cols(), TC.Q().rows(), tid);
            Index c_offset = 0;
            for (auto j = 0; j < TC.nSons(); ++j) {
              MMatrix temp3 =
                  acquireMap(TR.sons(i).Q().cols(), TC.sons(j).Q().cols(), tid);
              recursivelyComputeBlock_noalloc(TR.sons(i), TC.sons(j), e_gen,
                                              temp3, tid);
              const Index c_nscalfs = TC.sons(j).nscalfs();
              temp2.middleCols(c_offset, c_nscalfs) = temp3.leftCols(c_nscalfs);
              c_offset += c_nscalfs;
              releaseMap(temp3, tid);
            }
            const Index r_nscalfs = TR.sons(i).nscalfs();
            temp1.middleRows(r_offset, r_nscalfs).noalias() =
                temp2.topRows(r_nscalfs) * TC.Q();
            r_offset += r_nscalfs;
            releaseMap(temp2, tid);
          }
          block.noalias() = TR.Q().transpose() * temp1;
          releaseMap(temp1, tid);
          return;
        }
      }
    }
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  MemoryPool<Scalar> mem_arena_;
  MMatrix acquireMap(Index rows, Index cols, Index tid = 0) {
    return MMatrix(mem_arena_.acquire(rows * cols, tid), rows, cols);
    // return MMatrix(mem_arena_.acquire(tid), rows, cols);
  }

  void releaseMap(MMatrix &map, Index tid = 0) {
    if (!map.data()) return;
    mem_arena_.release(map.data(), map.rows() * map.cols(), tid);
    // mem_arena_.release(map.data(), tid);
    new (&map) MMatrix(nullptr, 0, 0);
  }
  DAG dag_;
  std::vector<typename DAG::Node *> sources_;
  std::ptrdiff_t max_size_;
};
}  // namespace FMCA

#endif

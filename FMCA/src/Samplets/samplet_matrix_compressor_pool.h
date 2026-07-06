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
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSORPOOL_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSORPOOL_H_

#include "../util/MemoryPool.h"
#include "../util/RandomTreeAccessor.h"

namespace FMCA {
namespace internal {

#if EIGEN_MAX_ALIGN_BYTES > 0
static_assert(MemoryPool<Scalar>::kAlign % EIGEN_MAX_ALIGN_BYTES == 0,
              "MemoryPool alignment must subsume Eigen's alignment");
#endif

/**
 *  \brief SampletMatrixCompressor with all matrix blocks and recursion
 *         scratch drawn from a thread-cached, size-class MemoryPool.
 *
 *  Every buffer is exactly sized (no worst-case slabs): stored blocks
 *  are pr->Q().cols() x pc->Q().cols(), recursion scratch is allocated
 *  and released LIFO per node so it always hits the thread-local
 *  freelist. Blocks are freed as soon as the parent level has consumed
 *  them; their triplets are emitted in the same pass. The pool is
 *  lifetime and thread agnostic (blocks may be freed by any thread), so
 *  the level loop can later be replaced by a task DAG with per-block
 *  consumer refcounts without touching the allocator.
 **/
template <typename Derived, typename ClusterComparison = CompareCluster>
class SampletMatrixCompressorPool {
 public:
  typedef std::map<size_t, Scalar *, std::greater<size_t>> LevelBuffer;
  // every pool block is 128-byte aligned by construction
  typedef Eigen::Map<Matrix, Eigen::Aligned128> AlignedMap;

  SampletMatrixCompressorPool() {}
  SampletMatrixCompressorPool(const SampletTreeBase<Derived> &ST, Scalar eta,
                              Scalar threshold = 0) {
    init(ST, eta, threshold);
  }

  const std::vector<LevelBuffer> &pattern() { return pattern_; };
  const RandomTreeAccessor<Derived> &rta() { return rta_; };
  size_t pool_footprint() const { return pool_.footprint(); }

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   **/
  void init(const SampletTreeBase<Derived> &ST, Scalar eta,
            Scalar threshold = 0) {
    eta_ = eta;
    threshold_ = threshold;
    npts_ = ST.block_size();
    rta_.init(ST, ST.block_size());
    pattern_.resize(2 * rta_.max_level() + 1);
#pragma omp parallel for schedule(dynamic)
    for (Index j = 0; j < rta_.nodes().size(); ++j) {
      const Derived *pc = rta_.nodes()[j];
      /*
       *  For the moment, the compression does not exploit inheritance
       *  relations in the column clusters. Thus, to obtain an NlogN
       *  algorithm, we have to exploit this at least in the row clusters.
       *  This is facilitated by starting a DFS for each column cluster.
       */
      std::vector<const Derived *> row_stack;
      row_stack.push_back(std::addressof(ST.derived()));
      while (row_stack.size()) {
        const Derived *pr = row_stack.back();
        row_stack.pop_back();
        for (auto i = 0; i < pr->nSons(); ++i)
          if (ClusterComparison::compare(pr->sons(i), *pc, eta) != LowRank)
            row_stack.push_back(std::addressof(pr->sons(i)));
        if (pc->block_id() >= pr->block_id()) {
          const size_t id =
              pr->block_id() + rta_.nodes().size() * pc->block_id();
#pragma omp critical
          pattern_[pc->level() + pr->level()].insert({id, nullptr});
        }
      }
    }
    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
    pool_.init();
    triplet_list_.clear();
    // the column cluster tree is traversed bottom up
    const auto &rclusters = rta_.nodes();
    const auto &cclusters = rta_.nodes();
    const auto nclusters = rta_.nodes().size();
    for (int ll = pattern_.size() - 1; ll >= 0; --ll) {
      Index pos = 0;
      const size_t map_size = pattern_[ll].size();
      LevelBuffer::iterator it2 = pattern_[ll].begin();
#pragma omp parallel shared(pos), firstprivate(it2)
      {
        Index i = 0;
        Index prev_i = 0;
#pragma omp atomic capture
        i = pos++;
        while (i < map_size) {
          std::advance(it2, i - prev_i);
          const Derived *pr = rclusters[it2->first % nclusters];
          const Derived *pc = cclusters[it2->first / nclusters];
          const Index col_id = pc->block_id();
          const Index row_id = pr->block_id();
          const Index brows = pr->Q().cols();
          const Index bcols = pc->Q().cols();
          Index nscalfs = 0;
          Index son_lvl = 0;
          Index offset = 0;
          size_t son_id = 0;
          const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());
          switch (the_case) {
            // (leaf,leaf), compute the block
            case 3: {
              it2->second = pool_.alloc(brows * bcols);
              recursivelyComputeBlock_noalloc(*pr, *pc, e_gen, it2->second);
              break;
            }
            // (noleaf,leaf), recycle from below
            case 1: {
              const Index qrows = pr->Q().rows();
              Scalar *bufm = pool_.alloc(qrows * bcols);
              AlignedMap buf(bufm, qrows, bcols);
              for (auto k = 0; k < pr->nSons(); ++k) {
                nscalfs = pr->sons(k).nscalfs();
                son_lvl = pr->sons(k).level() + pc->level();
                son_id = pr->sons(k).block_id() + nclusters * col_id;
                const auto it3 = pattern_[son_lvl].find(son_id);
                // if found, reuse the matrix block, otherwise recompute it
                if (it3 != pattern_[son_lvl].end()) {
                  AlignedMap ret(it3->second, pr->sons(k).Q().cols(), bcols);
                  buf.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                } else {
                  const Index n = pr->sons(k).Q().cols() * bcols;
                  Scalar *tmp = pool_.alloc(n);
                  recursivelyComputeBlock_noalloc(pr->sons(k), *pc, e_gen,
                                                  tmp);
                  AlignedMap ret(tmp, pr->sons(k).Q().cols(), bcols);
                  buf.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                  pool_.free(tmp, n);
                }
                offset += nscalfs;
              }
              it2->second = pool_.alloc(brows * bcols);
              AlignedMap retval(it2->second, brows, bcols);
              retval.noalias() = pr->Q().transpose() * buf;
              pool_.free(bufm, qrows * bcols);
              break;
            }
            // (*,noleaf), recycle from right
            case 2:
            case 0: {
              const Index qrows = pc->Q().rows();
              Scalar *bufm = pool_.alloc(brows * qrows);
              AlignedMap buf(bufm, brows, qrows);
              for (auto k = 0; k < pc->nSons(); ++k) {
                nscalfs = pc->sons(k).nscalfs();
                son_lvl = pc->sons(k).level() + pr->level();
                son_id = pc->sons(k).block_id() * nclusters + row_id;
                // check if pc's son is found in the row of pr
                // if found, reuse the matrix block, otherwise recompute it
                const auto it3 = pattern_[son_lvl].find(son_id);
                if (it3 != pattern_[son_lvl].end()) {
                  AlignedMap ret(it3->second, brows, pc->sons(k).Q().cols());
                  buf.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                } else {
                  const Index n = brows * pc->sons(k).Q().cols();
                  Scalar *tmp = pool_.alloc(n);
                  recursivelyComputeBlock_noalloc(*pr, pc->sons(k), e_gen,
                                                  tmp);
                  AlignedMap ret(tmp, brows, pc->sons(k).Q().cols());
                  buf.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                  pool_.free(tmp, n);
                }
                offset += nscalfs;
              }
              it2->second = pool_.alloc(brows * bcols);
              AlignedMap retval(it2->second, brows, bcols);
              retval.noalias() = buf * pc->Q();
              pool_.free(bufm, brows * qrows);
              break;
            }
          }
          prev_i = i;
#pragma omp atomic capture
          i = pos++;
        }
      }
      // garbage collector: the level above has been consumed, emit its
      // triplets and recycle its blocks
      if (ll < pattern_.size() - 1) {
        Index pos = 0;
        const size_t map_size = pattern_[ll + 1].size();
        LevelBuffer::iterator it2 = pattern_[ll + 1].begin();
#pragma omp parallel shared(pos), firstprivate(it2)
        {
          std::vector<Triplet> list;
          Index i = 0;
          Index prev_i = 0;
#pragma omp atomic capture
          i = pos++;
          while (i < map_size) {
            std::advance(it2, i - prev_i);
            const Derived *pr = rclusters[it2->first % nclusters];
            const Derived *pc = cclusters[it2->first / nclusters];
            storeBlockTriplets(list, pr, pc, it2->second);
            it2->second = nullptr;
            prev_i = i;
#pragma omp atomic capture
            i = pos++;
          }
#pragma omp critical
          triplet_list_.insert(triplet_list_.end(), list.begin(), list.end());
        }
      }
    }
    // the root x root block remains
    {
      std::vector<Triplet> list;
      for (auto &&it : pattern_[0]) {
        const Derived *pr = rclusters[it.first % nclusters];
        const Derived *pc = cclusters[it.first / nclusters];
        storeBlockTriplets(list, pr, pc, it.second);
        it.second = nullptr;
      }
      triplet_list_.insert(triplet_list_.end(), list.begin(), list.end());
    }
    pattern_.resize(0);
    return;
  }

  const std::vector<Triplet> &triplets() const { return triplet_list_; }

  std::vector<Triplet> release_triplets() {
    std::vector<Triplet> retval;
    std::swap(triplet_list_, retval);
    return retval;
  }

 private:
  /**
   *  \brief recursively computes for a given pair of row and column
   *         clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *         A^SigmaSigma]; the result is written into out, which must
   *         hold TR.Q().cols() * TC.Q().cols() scalars. All scratch is
   *         drawn LIFO from the pool (thread-local freelist hits).
   **/
  template <typename EntryGenerator>
  void recursivelyComputeBlock_noalloc(const Derived &TR, const Derived &TC,
                                       const EntryGenerator &e_gen,
                                       Scalar *out) {
    const Index orows = TR.Q().cols();
    const Index ocols = TC.Q().cols();
    AlignedMap retval(out, orows, ocols);
    // check for admissibility
    if (ClusterComparison::compare(TR, TC, eta_) == LowRank) {
      const Index nips = TR.V().rows();
      const Index dim = TR.bb().rows();
      const Index wsize = nips * nips + 2 * dim * nips;
      Scalar *S = pool_.alloc(nips * nips);
      Scalar *work = pool_.alloc(wsize);
      e_gen.interpolate_kernel_noalloc(TR, TC, S, work);
      pool_.free(work, wsize);
      Scalar *tmp = pool_.alloc(nips * ocols);
      AlignedMap Smap(S, nips, nips);
      AlignedMap temp(tmp, nips, ocols);
      temp.noalias() = Smap * TC.V();
      retval.noalias() = TR.V().transpose() * temp;
      pool_.free(tmp, nips * ocols);
      pool_.free(S, nips * nips);
      return;
    }
    const char the_case = 2 * (!TR.nSons()) + !TC.nSons();
    switch (the_case) {
      case 3: {
        // both are leafs: compute the block and return
        const Index brows = TR.block_size();
        const Index bcols = TC.block_size();
        Scalar *ker = pool_.alloc(brows * bcols);
        Scalar *tmp = pool_.alloc(brows * ocols);
        e_gen.compute_dense_block_noalloc(TR, TC, ker);
        AlignedMap K(ker, brows, bcols);
        AlignedMap temp(tmp, brows, ocols);
        temp.noalias() = K * TC.Q();
        retval.noalias() = TR.Q().transpose() * temp;
        pool_.free(tmp, brows * ocols);
        pool_.free(ker, brows * bcols);
        return;
      }
      case 2: {
        // the row cluster is a leaf cluster: recursion on the col cluster
        const Index qrows = TC.Q().rows();
        Scalar *bufm = pool_.alloc(orows * qrows);
        AlignedMap buf(bufm, orows, qrows);
        Index offset = 0;
        for (auto j = 0; j < TC.nSons(); ++j) {
          const Index n = orows * TC.sons(j).Q().cols();
          Scalar *tmp = pool_.alloc(n);
          recursivelyComputeBlock_noalloc(TR, TC.sons(j), e_gen, tmp);
          AlignedMap temp(tmp, orows, TC.sons(j).Q().cols());
          const Index nscalfs = TC.sons(j).nscalfs();
          buf.middleCols(offset, nscalfs) = temp.leftCols(nscalfs);
          offset += nscalfs;
          pool_.free(tmp, n);
        }
        retval.noalias() = buf * TC.Q();
        pool_.free(bufm, orows * qrows);
        return;
      }
      case 1: {
        // the col cluster is a leaf cluster: recursion on the row cluster
        const Index qrows = TR.Q().rows();
        Scalar *bufm = pool_.alloc(qrows * ocols);
        AlignedMap buf(bufm, qrows, ocols);
        Index offset = 0;
        for (auto i = 0; i < TR.nSons(); ++i) {
          const Index n = TR.sons(i).Q().cols() * ocols;
          Scalar *tmp = pool_.alloc(n);
          recursivelyComputeBlock_noalloc(TR.sons(i), TC, e_gen, tmp);
          AlignedMap temp(tmp, TR.sons(i).Q().cols(), ocols);
          const Index nscalfs = TR.sons(i).nscalfs();
          buf.middleRows(offset, nscalfs) = temp.topRows(nscalfs);
          offset += nscalfs;
          pool_.free(tmp, n);
        }
        retval.noalias() = TR.Q().transpose() * buf;
        pool_.free(bufm, qrows * ocols);
        return;
      }
      case 0: {
        // neither is a leaf, let recursion handle this
        const Index qrows = TR.Q().rows();
        Scalar *bufm = pool_.alloc(qrows * ocols);
        AlignedMap buf(bufm, qrows, ocols);
        Index r_offset = 0;
        for (auto i = 0; i < TR.nSons(); ++i) {
          const Index srows = TR.sons(i).Q().cols();
          const Index cqrows = TC.Q().rows();
          Scalar *cbufm = pool_.alloc(srows * cqrows);
          AlignedMap cbuf(cbufm, srows, cqrows);
          Index c_offset = 0;
          for (auto j = 0; j < TC.nSons(); ++j) {
            const Index n = srows * TC.sons(j).Q().cols();
            Scalar *tmp = pool_.alloc(n);
            recursivelyComputeBlock_noalloc(TR.sons(i), TC.sons(j), e_gen,
                                            tmp);
            AlignedMap temp(tmp, srows, TC.sons(j).Q().cols());
            const Index c_nscalfs = TC.sons(j).nscalfs();
            cbuf.middleCols(c_offset, c_nscalfs) = temp.leftCols(c_nscalfs);
            c_offset += c_nscalfs;
            pool_.free(tmp, n);
          }
          Scalar *resm = pool_.alloc(srows * ocols);
          AlignedMap res(resm, srows, ocols);
          res.noalias() = cbuf * TC.Q();
          const Index r_nscalfs = TR.sons(i).nscalfs();
          buf.middleRows(r_offset, r_nscalfs) = res.topRows(r_nscalfs);
          r_offset += r_nscalfs;
          pool_.free(resm, srows * ocols);
          pool_.free(cbufm, srows * cqrows);
        }
        retval.noalias() = TR.Q().transpose() * buf;
        pool_.free(bufm, qrows * ocols);
        return;
      }
    }
    return;
  }

  /**
   *  \brief emits the a-posteriori thresholded triplets of a finished
   *         block and recycles its memory
   **/
  void storeBlockTriplets(std::vector<Triplet> &list, const Derived *pr,
                          const Derived *pc, Scalar *block) {
    AlignedMap mat(block, pr->Q().cols(), pc->Q().cols());
    if (!pr->is_root() && !pc->is_root())
      storeBlock(list, pr->start_index(), pc->start_index(), pr->nsamplets(),
                 pc->nsamplets(),
                 mat.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
    else if (!pc->is_root())
      storeBlock(list, pr->start_index(), pc->start_index(), pr->Q().cols(),
                 pc->nsamplets(), mat.rightCols(pc->nsamplets()));
    else if (pr->is_root() && pc->is_root())
      storeBlock(list, pr->start_index(), pc->start_index(), pr->Q().cols(),
                 pc->Q().cols(), mat);
    pool_.free(block, pr->Q().cols() * pc->Q().cols());
    return;
  }

  /**
   *  \brief writes a given matrix block into a-posteriori thresholded
   *         triplet format
   **/
  template <typename otherDerived>
  void storeBlock(std::vector<Triplet> &triplet_buffer, Index srow, Index scol,
                  Index nrows, Index ncols,
                  const MatrixBase<otherDerived> &block) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if ((srow + j <= scol + k && std::abs(block(j, k)) > threshold_) ||
            (srow == scol && j == k))
          triplet_buffer.push_back(Triplet(srow + j, scol + k, block(j, k)));
  }
  //////////////////////////////////////////////////////////////////////////////
  MemoryPool<Scalar> pool_;
  std::vector<Triplet> triplet_list_;
  std::vector<LevelBuffer> pattern_;
  RandomTreeAccessor<Derived> rta_;
  Scalar eta_;
  Scalar threshold_;
  Index npts_;
};
}  // namespace internal
}  // namespace FMCA

#endif

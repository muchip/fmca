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

#include "../util/MemoryPool.h"
#include "../util/RandomTreeAccessor.h"

namespace FMCA {
namespace internal {
template <typename Derived, typename ClusterComparison = CompareCluster>
class SampletMatrixCompressor {
 public:
  typedef std::map<size_t, Scalar *, std::greater<size_t>> LevelBuffer;
  SampletMatrixCompressor() {}
  SampletMatrixCompressor(const SampletTreeBase<Derived> &ST, Scalar eta,
                          Scalar threshold = 0) {
    init(ST, eta, threshold);
  }

  const std::vector<LevelBuffer> &pattern() { return pattern_; };

  const RandomTreeAccessor<Derived> &rta() { return rta_; };

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   *
   **/
  void init(const SampletTreeBase<Derived> &ST, Scalar eta,
            Scalar threshold = 0) {
    eta_ = eta;
    threshold_ = threshold;
    npts_ = ST.block_size();
    rta_.init(ST, ST.block_size());
    pattern_.resize(2 * rta_.max_level() + 1);
    max_size_ = 0;
    std::vector<Index> block_sizes;
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
        // fill the stack with possible children
        for (auto i = 0; i < pr->nSons(); ++i)
          if (ClusterComparison::compare(pr->sons(i), *pc, eta) != LowRank)
            row_stack.push_back(std::addressof(pr->sons(i)));
        if (pc->block_id() >= pr->block_id()) {
          const size_t id =
              pr->block_id() + rta_.nodes().size() * pc->block_id();
#pragma omp critical
          {
            pattern_[pc->level() + pr->level()].insert({id, nullptr});
            max_size_ = std::max<std::ptrdiff_t>(
                {max_size_, pr->Q().rows(), pr->Q().cols(), pr->V().rows()});
            max_size_ = std::max<std::ptrdiff_t>(
                {max_size_, pc->Q().rows(), pc->Q().cols(), pc->V().rows()});
            block_sizes.push_back(pr->Q().cols() * pc->Q().cols());
          }
        }
      }
    }
    std::sort(block_sizes.begin(), block_sizes.end());
    std::cout << block_sizes.front() << "/" << block_sizes.back()
              << " median: " << block_sizes[block_sizes.size() / 2]
              << std::endl;
    {
      constexpr int kNumBins = 20;
      const double lo = std::log2(
          static_cast<double>(std::max<Index>(block_sizes.front(), 1)));
      const double hi = std::log2(static_cast<double>(block_sizes.back()));
      std::vector<std::size_t> bins(kNumBins, 0);
      for (Index sz : block_sizes) {
        const double t =
            (hi > lo)
                ? (std::log2(static_cast<double>(std::max<Index>(sz, 1))) -
                   lo) /
                      (hi - lo)
                : 0.0;
        const int b = std::min(kNumBins - 1, static_cast<int>(t * kNumBins));
        ++bins[b];
      }
      const std::size_t max_count = *std::max_element(bins.begin(), bins.end());
      constexpr int kBarWidth = 50;
      for (int b = 0; b < kNumBins; ++b) {
        const double lo_val = std::exp2(lo + b * (hi - lo) / kNumBins);
        const int bar_len =
            max_count
                ? static_cast<int>(kBarWidth * static_cast<double>(bins[b]) /
                                   max_count)
                : 0;
        std::cout << std::setw(8) << static_cast<Index>(lo_val) << " | "
                  << std::string(bar_len, '#') << " (" << bins[b] << ")"
                  << std::endl;
      }
    }
    std::cout << "determined maximum mem size:  " << max_size_ << std::endl;
    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
    const Index max_threads = omp_get_max_threads();
    triplet_list_.clear();
    std::vector<std::vector<Triplet>> tlist(max_threads);
    mem_arena_.init(max_size_, max_threads);
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
        const Index tid = omp_get_thread_num();
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
          Index nscalfs = 0;
          Index son_lvl = 0;
          Index offset = 0;
          size_t son_id = 0;
          Scalar *&block = it2->second;
          const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());
          switch (the_case) {
            // (leaf,leaf), compute the block
            case 3: {
              block = mem_arena_.acquire(tid);
              recursivelyComputeBlock_noalloc(*pr, *pc, e_gen, block, tid);
              break;
            }
            // (noleaf,leaf), recycle from below
            case 1: {
              block = mem_arena_.acquire(tid);
              Scalar *buf_mem = mem_arena_.acquire(tid);
              AMap<Matrix> buf(buf_mem, pr->Q().rows(), pc->Q().cols());
              for (auto k = 0; k < pr->nSons(); ++k) {
                nscalfs = pr->sons(k).nscalfs();
                son_lvl = pr->sons(k).level() + pc->level();
                son_id = pr->sons(k).block_id() + nclusters * col_id;
                const auto it3 = pattern_[son_lvl].find(son_id);
                // if so, reuse the matrix block, otherwise recompute it
                if (it3 != pattern_[son_lvl].end()) {
                  AMap<Matrix> ret(it3->second, pr->sons(k).Q().cols(),
                                   pc->Q().cols());
                  buf.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                } else {
                  Scalar *mem = mem_arena_.acquire(tid);
                  recursivelyComputeBlock_noalloc(pr->sons(k), *pc, e_gen, mem,
                                                  tid);
                  AMap<Matrix> ret(mem, pr->sons(k).Q().cols(), pc->Q().cols());
                  buf.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                  mem_arena_.release(mem, tid);
                }
                offset += nscalfs;
              }
              AMap<Matrix> retval(block, pr->Q().cols(), pc->Q().cols());
              retval.noalias() = pr->Q().transpose() * buf;
              mem_arena_.release(buf_mem, tid);
              break;
            }
              // (*,noleaf), recycle from right
            case 2:
            case 0: {
              block = mem_arena_.acquire(tid);
              Scalar *buf_mem = mem_arena_.acquire(tid);
              AMap<Matrix> buf(buf_mem, pr->Q().cols(), pc->Q().rows());
              for (auto k = 0; k < pc->nSons(); ++k) {
                nscalfs = pc->sons(k).nscalfs();
                son_lvl = pc->sons(k).level() + pr->level();
                son_id = pc->sons(k).block_id() * nclusters + row_id;
                // check if pc's son is found in the row of pr
                // if so, reuse the matrix block, otherwise recompute it
                const auto it3 = pattern_[son_lvl].find(son_id);
                if (it3 != pattern_[son_lvl].end()) {
                  AMap<Matrix> ret(it3->second, pr->Q().cols(),
                                   pc->sons(k).Q().cols());
                  buf.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                } else {
                  Scalar *mem = mem_arena_.acquire(tid);
                  recursivelyComputeBlock_noalloc(*pr, pc->sons(k), e_gen, mem,
                                                  tid);
                  AMap<Matrix> ret(mem, pr->Q().cols(), pc->sons(k).Q().cols());
                  buf.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                  mem_arena_.release(mem, tid);
                }
                offset += nscalfs;
              }
              AMap<Matrix> retval(block, pr->Q().cols(), pc->Q().cols());
              retval.noalias() = buf * pc->Q();
              mem_arena_.release(buf_mem, tid);
              break;
            }
          }
          // tag_[row_id].insert(col_id);
          prev_i = i;
#pragma omp atomic capture
          i = pos++;
        }
      }
      // garbage collector
      if (ll < pattern_.size() - 1) {
        Index pos = 0;
        const size_t map_size = pattern_[ll + 1].size();
        LevelBuffer::iterator it2 = pattern_[ll + 1].begin();
#pragma omp parallel shared(pos), firstprivate(it2)
        {
          const Index tid = omp_get_thread_num();
          Index i = 0;
          Index prev_i = 0;
#pragma omp atomic capture
          i = pos++;
          while (i < map_size) {
            std::advance(it2, i - prev_i);
            const Derived *pr = rclusters[it2->first % nclusters];
            const Derived *pc = cclusters[it2->first / nclusters];
            Scalar *&block = it2->second;
            AMap<Matrix> mat(block, pr->Q().cols(), pc->Q().cols());
            if (!pr->is_root() && !pc->is_root())
              storeBlock(
                  tlist[tid], pr->start_index(), pc->start_index(),
                  pr->nsamplets(), pc->nsamplets(),
                  mat.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
            else if (!pc->is_root())
              storeBlock(tlist[tid], pr->start_index(), pc->start_index(),
                         pr->Q().cols(), pc->nsamplets(),
                         mat.rightCols(pc->nsamplets()));
            else if (pr->is_root() && pc->is_root())
              storeBlock(tlist[tid], pr->start_index(), pc->start_index(),
                         pr->Q().cols(), pc->Q().cols(), mat);
            mem_arena_.release(block, tid);
            block = nullptr;
            prev_i = i;
#pragma omp atomic capture
            i = pos++;
          }
        }
      }
    }
    // garbage collector
    {
      Index pos = 0;
      const size_t map_size = pattern_[0].size();
      LevelBuffer::iterator it2 = pattern_[0].begin();
      {
        Index i = 0;
        Index prev_i = 0;
        i = pos++;
        while (i < map_size) {
          std::advance(it2, i - prev_i);
          const Derived *pr = rclusters[it2->first % nclusters];
          const Derived *pc = cclusters[it2->first / nclusters];
          Scalar *&block = it2->second;
          AMap<Matrix> mat(block, pr->Q().cols(), pc->Q().cols());
          if (!pr->is_root() && !pc->is_root())
            storeBlock(tlist[0], pr->start_index(), pc->start_index(),
                       pr->nsamplets(), pc->nsamplets(),
                       mat.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
          else if (!pc->is_root())
            storeBlock(tlist[0], pr->start_index(), pc->start_index(),
                       pr->Q().cols(), pc->nsamplets(),
                       mat.rightCols(pc->nsamplets()));
          else if (pr->is_root() && pc->is_root())
            storeBlock(tlist[0], pr->start_index(), pc->start_index(),
                       pr->Q().cols(), pc->Q().cols(), mat);
          mem_arena_.release(block, 0);
          block = nullptr;
          prev_i = i;
          i = pos++;
        }
      }
    }
    for (Index i = 0; i < tlist.size(); ++i)
      triplet_list_.insert(triplet_list_.end(), tlist[i].begin(),
                           tlist[i].end());
    return;
  }

  /**
   *  \brief creates a posteriori thresholded triplets and stores them to in
   *the triplet list
   **/
  std::vector<Triplet> a_priori_pattern_triplets() {
    std::vector<Triplet> retval;
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < pattern_.size(); ++i) {
      std::vector<Triplet> list;
      for (auto &&it : pattern_[i]) {
        const Derived *pr = rta_.nodes()[it.first % rta_.nodes().size()];
        const Derived *pc = rta_.nodes()[it.first / rta_.nodes().size()];
        if (!pr->is_root() && !pc->is_root())
          storeEmptyBlock(list, pr->start_index(), pc->start_index(),
                          pr->nsamplets(), pc->nsamplets());
        else if (!pc->is_root())
          storeEmptyBlock(list, pr->start_index(), pc->start_index(),
                          pr->Q().cols(), pc->nsamplets());
        else if (pr->is_root() && pc->is_root())
          storeEmptyBlock(list, pr->start_index(), pc->start_index(),
                          pr->Q().cols(), pc->Q().cols());
      }
#pragma omp critical
      retval.insert(retval.end(), list.begin(), list.end());
    }
    return retval;
  }



 private:
  /**
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  void recursivelyComputeBlock_noalloc(const Derived &TR, const Derived &TC,
                                       const EntryGenerator &e_gen, Scalar *mem,
                                       Index tid = 0) {
    // check for admissibility
    if (ClusterComparison::compare(TR, TC, eta_) == LowRank) {
      Scalar *temp_mem = mem_arena_.acquire(tid);
      e_gen.interpolate_kernel_noalloc(TR, TC, mem, temp_mem);
      AMap<Matrix> buf(mem, TR.V().rows(), TC.V().rows());
      AMap<Matrix> temp(temp_mem, TR.V().rows(), TC.V().cols());
      temp.noalias() = buf * TC.V();
      AMap<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
      retval.noalias() = TR.V().transpose() * temp;
      mem_arena_.release(temp_mem, tid);
      return;
    } else {
      const char the_case = 2 * (!TR.nSons()) + !TC.nSons();
      switch (the_case) {
        case 3: {
          // both are leafs: compute the block and return
          Scalar *temp_mem = mem_arena_.acquire(tid);
          e_gen.compute_dense_block_noalloc(TR, TC, mem);
          AMap<Matrix> buf(mem, TR.Q().rows(), TC.Q().rows());
          AMap<Matrix> temp(temp_mem, TR.Q().rows(), TC.Q().cols());
          temp.noalias() = buf * TC.Q();
          AMap<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = TR.Q().transpose() * temp;
          mem_arena_.release(temp_mem, tid);
          return;
        }
        case 2: {
          Scalar *temp_mem = mem_arena_.acquire(tid);
          AMap<Matrix> buf(temp_mem, TR.Q().cols(), TC.Q().rows());
          // the row cluster is a leaf cluster: recursion on the col cluster
          Index offset = 0;
          for (auto j = 0; j < TC.nSons(); ++j) {
            recursivelyComputeBlock_noalloc(TR, TC.sons(j), e_gen, mem, tid);
            AMap<Matrix> temp(mem, TR.Q().cols(), TC.sons(j).Q().cols());
            const Index nscalfs = TC.sons(j).nscalfs();
            buf.middleCols(offset, nscalfs) = temp.leftCols(nscalfs);
            offset += nscalfs;
          }
          AMap<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = buf * TC.Q();
          mem_arena_.release(temp_mem, tid);
          return;
        }
        case 1: {
          Scalar *temp_mem = mem_arena_.acquire(tid);
          AMap<Matrix> buf(temp_mem, TR.Q().rows(), TC.Q().cols());
          // the col cluster is a leaf cluster: recursion on the row cluster
          Index offset = 0;
          for (auto i = 0; i < TR.nSons(); ++i) {
            recursivelyComputeBlock_noalloc(TR.sons(i), TC, e_gen, mem, tid);
            AMap<Matrix> temp(mem, TR.sons(i).Q().cols(), TC.Q().cols());
            const Index nscalfs = TR.sons(i).nscalfs();
            buf.middleRows(offset, nscalfs) = temp.topRows(nscalfs);
            offset += nscalfs;
          }
          AMap<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = TR.Q().transpose() * buf;
          mem_arena_.release(temp_mem, tid);
          return;
        }
        case 0: {
          Scalar *temp_mem = mem_arena_.acquire(tid);
          AMap<Matrix> buf(temp_mem, TR.Q().rows(), TC.Q().cols());
          // neither is a leaf, let recursion handle this
          Index r_offset = 0;
          Scalar *r_mem = mem_arena_.acquire(tid);
          for (auto i = 0; i < TR.nSons(); ++i) {
            AMap<Matrix> cbuf(mem, TR.sons(i).Q().cols(), TC.Q().rows());
            Index c_offset = 0;
            for (auto j = 0; j < TC.nSons(); ++j) {
              recursivelyComputeBlock_noalloc(TR.sons(i), TC.sons(j), e_gen,
                                              r_mem, tid);
              AMap<Matrix> temp(r_mem, TR.sons(i).Q().cols(),
                                TC.sons(j).Q().cols());
              const Index c_nscalfs = TC.sons(j).nscalfs();
              cbuf.middleCols(c_offset, c_nscalfs) = temp.leftCols(c_nscalfs);
              c_offset += c_nscalfs;
            }
            AMap<Matrix> res_buf(r_mem, TR.sons(i).Q().cols(), TC.Q().cols());
            res_buf.noalias() = cbuf * TC.Q();
            const Index r_nscalfs = TR.sons(i).nscalfs();
            buf.middleRows(r_offset, r_nscalfs) = res_buf.topRows(r_nscalfs);
            r_offset += r_nscalfs;
          }
          mem_arena_.release(r_mem, tid);
          AMap<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = TR.Q().transpose() * buf;
          mem_arena_.release(temp_mem, tid);
          return;
        }
      }
    }
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

  void storeEmptyBlock(std::vector<Triplet> &triplet_buffer, Index srow,
                       Index scol, Index nrows, Index ncols) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (srow + j <= scol + k)
          triplet_buffer.push_back(Triplet(srow + j, scol + k, 0));
  }
  //////////////////////////////////////////////////////////////////////////////
  MemoryPool<Scalar> mem_arena_;
  std::vector<Triplet> triplet_list_;
  std::vector<LevelBuffer> pattern_;
  RandomTreeAccessor<Derived> rta_;
  std::ptrdiff_t max_size_;
  Scalar eta_;
  Scalar threshold_;
  Index npts_;
};
}  // namespace internal
}  // namespace FMCA

#endif

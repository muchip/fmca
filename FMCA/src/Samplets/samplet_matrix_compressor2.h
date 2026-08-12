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

#include "../util/MemoryPool2.h"
#include "../util/RandomTreeAccessor.h"

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
  typedef std::map<size_t, MMatrix, std::greater<size_t>> LevelBuffer;
  SampletMatrixCompressor() {}
  SampletMatrixCompressor(const SampletTreeBase<H2STreeType> &ST, Scalar eta,
                          Scalar threshold = 0) {
    init(ST, eta, threshold);
  }

  const std::vector<LevelBuffer> &pattern() { return pattern_; };

  const internal::RandomTreeAccessor<H2STreeType> &rta() { return rta_; };

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   *
   **/
  void init(const SampletTreeBase<H2STreeType> &ST, Scalar eta,
            Scalar threshold = 0) {
    std::cout << "using compressor 2" << std::endl;
    Base::setDimensions(ST.block_size(), ST.block_size());
    Base::setThreshold(threshold);
    Base::setEta(eta);
    rta_.init(ST, ST.block_size());
    pattern_.resize(2 * rta_.max_level() + 1);
    max_size_ = 0;
    std::vector<Index> block_sizes;
#pragma omp parallel for schedule(dynamic)
    for (Index j = 0; j < rta_.nodes().size(); ++j) {
      const H2STreeType *pc = rta_.nodes()[j];
      /*
       *  For the moment, the compression does not exploit inheritance
       *  relations in the column clusters. Thus, to obtain an NlogN
       *  algorithm, we have to exploit this at least in the row clusters.
       *  This is facilitated by starting a DFS for each column cluster.
       */
      std::vector<const H2STreeType *> row_stack;
      row_stack.push_back(std::addressof(ST.derived()));
      while (row_stack.size()) {
        const H2STreeType *pr = row_stack.back();
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
            pattern_[pc->level() + pr->level()].insert(
                {id, MMatrix(nullptr, 0, 0)});
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
    if (false) {
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
    std::vector<std::vector<Triplet>> tlist(max_threads);
    mem_arena_.init(max_size_ * max_size_, max_threads);
    Base::clearTriplets();
    // mem_arena_.init(max_size_, max_threads);
    //  the column cluster tree is traversed bottom up
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
          const H2STreeType *pr = rclusters[it2->first % nclusters];
          const H2STreeType *pc = cclusters[it2->first / nclusters];
          const Index col_id = pc->block_id();
          const Index row_id = pr->block_id();
          Index nscalfs = 0;
          Index son_lvl = 0;
          Index offset = 0;
          size_t son_id = 0;
          MMatrix &block = it2->second;
          new (&block) MMatrix(acquireMap(pr->Q().cols(), pc->Q().cols(), tid));
          const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());
          switch (the_case) {
            // (leaf,leaf), compute the block
            case 3: {
              recursivelyComputeBlock_noalloc(*pr, *pc, e_gen, block, tid);
              break;
            }
            // (noleaf,leaf), recycle from below
            case 1: {
              MMatrix buf = acquireMap(pr->Q().rows(), pc->Q().cols(), tid);
              for (auto k = 0; k < pr->nSons(); ++k) {
                nscalfs = pr->sons(k).nscalfs();
                son_lvl = pr->sons(k).level() + pc->level();
                son_id = pr->sons(k).block_id() + nclusters * col_id;
                const auto it3 = pattern_[son_lvl].find(son_id);
                // if so, reuse the matrix block, otherwise recompute it
                if (it3 != pattern_[son_lvl].end()) {
                  const MMatrix &ret = it3->second;
                  buf.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                } else {
                  MMatrix temp =
                      acquireMap(pr->sons(k).Q().cols(), pc->Q().cols(), tid);
                  recursivelyComputeBlock_noalloc(pr->sons(k), *pc, e_gen, temp,
                                                  tid);
                  buf.middleRows(offset, nscalfs) = temp.topRows(nscalfs);
                  releaseMap(temp, tid);
                }
                offset += nscalfs;
              }
              block.noalias() = pr->Q().transpose() * buf;
              releaseMap(buf, tid);
              break;
            }
              // (*,noleaf), recycle from right
            case 2:
            case 0: {
              MMatrix buf = acquireMap(pr->Q().cols(), pc->Q().rows(), tid);
              for (auto k = 0; k < pc->nSons(); ++k) {
                nscalfs = pc->sons(k).nscalfs();
                son_lvl = pc->sons(k).level() + pr->level();
                son_id = pc->sons(k).block_id() * nclusters + row_id;
                // check if pc's son is found in the row of pr
                // if so, reuse the matrix block, otherwise recompute it
                const auto it3 = pattern_[son_lvl].find(son_id);
                if (it3 != pattern_[son_lvl].end()) {
                  const MMatrix &ret = it3->second;
                  buf.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                } else {
                  MMatrix temp =
                      acquireMap(pr->Q().cols(), pc->sons(k).Q().cols(), tid);
                  recursivelyComputeBlock_noalloc(*pr, pc->sons(k), e_gen, temp,
                                                  tid);
                  buf.middleCols(offset, nscalfs) = temp.leftCols(nscalfs);
                  releaseMap(temp, tid);
                }
                offset += nscalfs;
              }
              block.noalias() = buf * pc->Q();
              releaseMap(buf, tid);
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
            const H2STreeType *pr = rclusters[it2->first % nclusters];
            const H2STreeType *pc = cclusters[it2->first / nclusters];
            MMatrix &mat = it2->second;
            if (!pr->is_root() && !pc->is_root())
              storeSymBlock(
                  tlist[tid], pr->start_index(), pc->start_index(),
                  pr->nsamplets(), pc->nsamplets(),
                  mat.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
            else if (!pc->is_root())
              storeSymBlock(tlist[tid], pr->start_index(), pc->start_index(),
                            pr->Q().cols(), pc->nsamplets(),
                            mat.rightCols(pc->nsamplets()));
            else if (pr->is_root() && pc->is_root())
              storeSymBlock(tlist[tid], pr->start_index(), pc->start_index(),
                            pr->Q().cols(), pc->Q().cols(), mat);
            releaseMap(mat, tid);
            new (&mat) MMatrix(nullptr, 0, 0);
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
          const H2STreeType *pr = rclusters[it2->first % nclusters];
          const H2STreeType *pc = cclusters[it2->first / nclusters];
          MMatrix &mat = it2->second;
          if (!pr->is_root() && !pc->is_root())
            storeSymBlock(
                tlist[0], pr->start_index(), pc->start_index(), pr->nsamplets(),
                pc->nsamplets(),
                mat.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
          else if (!pc->is_root())
            storeSymBlock(tlist[0], pr->start_index(), pc->start_index(),
                          pr->Q().cols(), pc->nsamplets(),
                          mat.rightCols(pc->nsamplets()));
          else if (pr->is_root() && pc->is_root())
            storeSymBlock(tlist[0], pr->start_index(), pc->start_index(),
                          pr->Q().cols(), pc->Q().cols(), mat);
          releaseMap(mat, 0);
          new (&mat) MMatrix(nullptr, 0, 0);
          prev_i = i;
          i = pos++;
        }
      }
    }
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
                (temp2 * TC.Q()).topRows(r_nscalfs);
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
  std::vector<LevelBuffer> pattern_;
  internal::RandomTreeAccessor<H2STreeType> rta_;
  std::ptrdiff_t max_size_;
};
}  // namespace FMCA

#endif

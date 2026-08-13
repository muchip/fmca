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

#include "../util/DummyMemoryPool.h"
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

  typedef std::map<size_t, Matrix, std::greater<size_t>> LevelBuffer;
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
    std::cout << "using compressor 1" << std::endl;
    Base::setDimensions(ST.block_size(), ST.block_size());
    Base::setThreshold(threshold);
    Base::setEta(eta);
    rta_.init(ST, ST.block_size());
    pattern_.resize(2 * rta_.max_level() + 1);
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
          pattern_[pc->level() + pr->level()].insert({id, Matrix(0, 0)});
        }
      }
    }
    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
    const Index max_threads = omp_get_max_threads();
    std::vector<std::vector<Triplet>> tlist(max_threads);
    Base::clearTriplets();
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
          const H2STreeType *pr = rclusters[it2->first % nclusters];
          const H2STreeType *pc = cclusters[it2->first / nclusters];
          const Index col_id = pc->block_id();
          const Index row_id = pr->block_id();
          Index nscalfs = 0;
          Index son_lvl = 0;
          Index offset = 0;
          size_t son_id = 0;
          Matrix &block = it2->second;
          const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());
          switch (the_case) {
            // (leaf,leaf), compute the block
            case 3:
              block = recursivelyComputeBlock(*pr, *pc, e_gen);
              break;
            // (noleaf,leaf), recycle from below
            case 1:
              block.resize(pr->Q().rows(), pc->Q().cols());
              for (auto k = 0; k < pr->nSons(); ++k) {
                nscalfs = pr->sons(k).nscalfs();
                son_lvl = pr->sons(k).level() + pc->level();
                son_id = pr->sons(k).block_id() + nclusters * col_id;
                const auto it3 = pattern_[son_lvl].find(son_id);
                // if so, reuse the matrix block, otherwise recompute it
                if (it3 != pattern_[son_lvl].end()) {
                  const Matrix &ret = it3->second;
                  block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                } else {
                  const Matrix ret =
                      recursivelyComputeBlock(pr->sons(k), *pc, e_gen);
                  block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                }
                offset += nscalfs;
              }
              block = pr->Q().transpose() * block;
              break;
              // (*,noleaf), recycle from right
            case 2:
            case 0:
              block.resize(pr->Q().cols(), pc->Q().rows());
              for (auto k = 0; k < pc->nSons(); ++k) {
                nscalfs = pc->sons(k).nscalfs();
                son_lvl = pc->sons(k).level() + pr->level();
                son_id = pc->sons(k).block_id() * nclusters + row_id;
                // check if pc's son is found in the row of pr
                // if so, reuse the matrix block, otherwise recompute it
                const auto it3 = pattern_[son_lvl].find(son_id);
                if (it3 != pattern_[son_lvl].end()) {
                  const Matrix &ret = it3->second;
                  block.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                } else {
                  const Matrix ret =
                      recursivelyComputeBlock(*pr, pc->sons(k), e_gen);
                  block.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                }
                offset += nscalfs;
              }
              block = block * pc->Q();
              break;
          }
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
            Matrix &mat = it2->second;
            storeBlock(*pr, *pc, tlist[tid], mat);
            mat.resize(0, 0);
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
          Matrix &mat = it2->second;
          storeBlock(*pr, *pc, tlist[0], mat);
          mat.resize(0, 0);
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
  inline void storeBlock(const H2STreeType &TR, const H2STreeType &TC,
                         std::vector<Triplet> &triplet_buffer, Matrix &block) {
    const Index nrows = TR.is_root() ? TR.Q().cols() : TR.nsamplets();
    const Index ncols = TC.is_root() ? TC.Q().cols() : TC.nsamplets();
    Base::storeSymTriplets(triplet_buffer, TR.start_index(), TC.start_index(),
                           nrows, ncols, block.bottomRightCorner(nrows, ncols));
  }
  /**
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  Matrix recursivelyComputeBlock(const H2STreeType &TR, const H2STreeType &TC,
                                 const EntryGenerator &e_gen) {
    Matrix buf(0, 0);
    Index r_offset = 0;
    Index c_offset = 0;
    // check for admissibility
    if (ClusterComparison::compare(TR, TC, Base::eta()) == LowRank) {
      e_gen.interpolate_kernel(TR, TC, &buf);
      return TR.V().transpose() * buf * TC.V();
    } else {
      const char the_case = 2 * (!TR.nSons()) + !TC.nSons();
      switch (the_case) {
        case 3:
          // both are leafs: compute the block and return
          e_gen.compute_dense_block(TR, TC, &buf);
          return TR.Q().transpose() * buf * TC.Q();
        case 2:
          // the row cluster is a leaf cluster: recursion on the col cluster
          buf.resize(TR.Q().cols(), TC.Q().rows());
          c_offset = 0;
          for (Index j = 0; j < TC.nSons(); ++j) {
            const Index nscalfs = TC.sons(j).nscalfs();
            const Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
            buf.middleCols(c_offset, nscalfs) = ret.leftCols(nscalfs);
            c_offset += nscalfs;
          }
          return buf * TC.Q();
        case 1:
          // the col cluster is a leaf cluster: recursion on the row cluster
          buf.resize(TR.Q().rows(), TC.Q().cols());
          r_offset = 0;
          for (Index i = 0; i < TR.nSons(); ++i) {
            const Index nscalfs = TR.sons(i).nscalfs();
            const Matrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
            buf.middleRows(r_offset, nscalfs) = ret.topRows(nscalfs);
            r_offset += nscalfs;
          }
          return TR.Q().transpose() * buf;
        case 0:
          // neither is a leaf, let recursion handle this
          buf.resize(TR.Q().rows(), TC.Q().cols());
          r_offset = 0;
          for (Index i = 0; i < TR.nSons(); ++i) {
            Matrix buf2(TR.sons(i).Q().cols(), TC.Q().rows());
            c_offset = 0;
            const Index r_nscalfs = TR.sons(i).nscalfs();
            for (Index j = 0; j < TC.nSons(); ++j) {
              const Index c_nscalfs = TC.sons(j).nscalfs();
              const Matrix ret =
                  recursivelyComputeBlock(TR.sons(i), TC.sons(j), e_gen);
              buf2.middleCols(c_offset, c_nscalfs) = ret.leftCols(c_nscalfs);
              c_offset += c_nscalfs;
            }
            buf.middleRows(r_offset, r_nscalfs).noalias() =
                buf2.topRows(r_nscalfs) * TC.Q();
            r_offset += r_nscalfs;
          }
          return TR.Q().transpose() * buf;
      }
    }
    return Matrix(0, 0);
  }

  //////////////////////////////////////////////////////////////////////////////
  std::vector<LevelBuffer> pattern_;
  internal::RandomTreeAccessor<H2STreeType> rta_;
};
}  // namespace FMCA

#endif

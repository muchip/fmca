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
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSOR_UNSYMMETRIC_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXCOMPRESSOR_UNSYMMETRIC_H_

#include "../util/RandomTreeAccessor.h"
#include "recursivelyComputeBlock.h"

namespace FMCA {
template <typename H2STreeType, typename ClusterComparison = CompareCluster>
class SampletMatrixCompressorUnsymmetric
    : public SampletMatrixCompressorBase<
          SampletMatrixCompressorUnsymmetric<H2STreeType, ClusterComparison>> {
 public:
  typedef SampletMatrixCompressorBase<
      SampletMatrixCompressorUnsymmetric<H2STreeType, ClusterComparison>>
      Base;
  using Base::aposteriori_triplets;
  using Base::aposteriori_triplets_fast;
  using Base::cols;
  using Base::rows;
  using Base::triplets;

  typedef std::map<size_t, Matrix, std::greater<size_t>> LevelBuffer;
  SampletMatrixCompressorUnsymmetric() {}
  SampletMatrixCompressorUnsymmetric(const SampletTreeBase<H2STreeType> &TR,
                                     const SampletTreeBase<H2STreeType> &TC,
                                     Scalar eta, Scalar threshold = 0) {
    init(TR, TC, eta, threshold);
  }
  const std::vector<LevelBuffer> &pattern() { return pattern_; };

  const internal::RandomTreeAccessor<H2STreeType> &crta() { return c_rta_; };
  const internal::RandomTreeAccessor<H2STreeType> &rrta() { return r_rta_; };

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   *
   **/
  void init(const SampletTreeBase<H2STreeType> &TR,
            const SampletTreeBase<H2STreeType> &TC, Scalar eta,
            Scalar threshold = 0) {
    std::cout << "using unsymmetric compressor 1" << std::endl;
    Base::setDimensions(TR.block_size(), TC.block_size());
    Base::setThreshold(threshold);
    Base::setEta(eta);
    r_rta_.init(TR, TR.block_size());
    c_rta_.init(TC, TC.block_size());
    pattern_.resize(c_rta_.nodes().size());
    queue_.resize(r_rta_.max_level() + c_rta_.max_level() + 1);
    for (Index j = 0; j < c_rta_.nodes().size(); ++j) {
      const H2STreeType *pc = c_rta_.nodes()[j];
      /*
       *  For the moment, the compression does not exploit inheritance
       *  relations in the column clusters. Thus, to obtain an NlogN
       *  algorithm, we have to exploit this at least in the row clusters.
       *  This is facilitated by starting a DFS for each column cluster.
       */
      std::vector<const H2STreeType *> row_stack;
      row_stack.push_back(std::addressof(TR.derived()));
      while (row_stack.size()) {
        const H2STreeType *pr = row_stack.back();
        row_stack.pop_back();
        // fill the stack with possible children
        for (auto i = 0; i < pr->nSons(); ++i)
          if (ClusterComparison::compare(pr->sons(i), *pc, eta) != LowRank)
            row_stack.push_back(std::addressof(pr->sons(i)));
        auto it =
            pattern_[pc->block_id()].insert({pr->block_id(), Matrix(0, 0)});
        queue_[pc->level() + pr->level()].push_back(
            ijp(pr->block_id(), pc->block_id(),
                std::addressof((it.first)->second)));
      }
    }
    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
    const Index max_threads = omp_get_max_threads();
    std::vector<std::vector<Triplet>> tlist(max_threads);
    // the column cluster tree is traversed bottom up
    Base::clearTriplets();
    const auto &rclusters = r_rta_.nodes();
    const auto &cclusters = c_rta_.nodes();
    for (auto it = queue_.rbegin(); it != queue_.rend(); ++it) {
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < it->size(); ++i) {
        const H2STreeType *pr = rclusters[(*it)[i].i];
        const H2STreeType *pc = cclusters[(*it)[i].j];
        Matrix &block = *((*it)[i].p);
        const Index col_id = pc->block_id();
        const Index row_id = pr->block_id();
        block.resize(0, 0);
        //  preferred, we pick blocks from the right
        if (pc->nSons()) {
          for (auto k = 0; k < pc->nSons(); ++k) {
            const Index nscalfs = pc->sons(k).nscalfs();
            const Index son_id = pc->sons(k).block_id();
            // check if pc's son is found in the row of pr
            // if so, reuse the matrix block, otherwise recompute it
            const auto it3 = pattern_[son_id].find(row_id);
            if (it3 != pattern_[son_id].end()) {
              const Matrix &ret = it3->second;
              block.conservativeResize(ret.rows(), block.cols() + nscalfs);
              block.rightCols(nscalfs) = ret.leftCols(nscalfs);
            } else {
              const Matrix ret = computeBlock(*pr, pc->sons(k), e_gen);
              block.conservativeResize(ret.rows(), block.cols() + nscalfs);
              block.rightCols(nscalfs) = ret.leftCols(nscalfs);
            }
          }
          block = block * pc->Q();
        } else {
          if (!pr->nSons()) {
            block = computeBlock(*pr, *pc, e_gen);
          } else {
            for (auto k = 0; k < pr->nSons(); ++k) {
              const Index nscalfs = pr->sons(k).nscalfs();
              const Index son_id = pr->sons(k).block_id();
              const auto it3 = pattern_[col_id].find(son_id);
              // if so, reuse the matrix block, otherwise recompute it
              if (it3 != pattern_[col_id].end()) {
                const Matrix &ret = it3->second;
                block.conservativeResize(ret.cols(), block.cols() + nscalfs);
                block.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
              } else {
                const Matrix ret = computeBlock(pr->sons(k), *pc, e_gen);
                block.conservativeResize(ret.cols(), block.cols() + nscalfs);
                block.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
              }
            }
            block = (block * pr->Q()).transpose();
          }
        }
      }
      // garbage collector
      if (it != queue_.rbegin()) {
        auto itm1 = it;
        --itm1;
#pragma omp parallel for
        for (Index i = 0; i < itm1->size(); ++i) {
          const Index tid = omp_get_thread_num();
          const H2STreeType *pr = rclusters[(*itm1)[i].i];
          const H2STreeType *pc = cclusters[(*itm1)[i].j];
          Matrix &block = *((*itm1)[i].p);
          storeBlock(*pr, *pc, tlist[tid], block);
          block.resize(0, 0);
        }
      }
    }
    // garbage collector
    {
      auto itm1 = queue_.begin();
      for (Index i = 0; i < itm1->size(); ++i) {
        const H2STreeType *pr = rclusters[(*itm1)[i].i];
        const H2STreeType *pc = cclusters[(*itm1)[i].j];
        Matrix &block = *((*itm1)[i].p);
        storeBlock(*pr, *pc, tlist[0], block);
        block.resize(0, 0);
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
    Base::storeTriplets(triplet_buffer, TR.start_index(), TC.start_index(),
                        nrows, ncols, block.bottomRightCorner(nrows, ncols));
  }
  template <typename EntGen>
  Matrix computeBlock(const H2STreeType &TR, const H2STreeType &TC,
                      const EntGen &e_gen) const {
    return internal::recursivelyComputeBlock<H2STreeType, EntGen,
                                             ClusterComparison>(TR, TC, e_gen,
                                                                Base::eta());
  }

  //////////////////////////////////////////////////////////////////////////////
  struct ijp {
    Index i;
    Index j;
    Matrix *p;
    ijp(Index ii, Index jj, Matrix *pp) : i(ii), j(jj), p(pp) {};
  };
  std::vector<std::vector<ijp>> queue_;
  std::vector<LevelBuffer> pattern_;
  internal::RandomTreeAccessor<H2STreeType> r_rta_;
  internal::RandomTreeAccessor<H2STreeType> c_rta_;
};
}  // namespace FMCA

#endif

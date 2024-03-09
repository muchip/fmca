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
#ifndef FMCA_SAMPLETS_MPISAMPLETMATRIXCOMPRESSOR_H_
#define FMCA_SAMPLETS_MPISAMPLETMATRIXCOMPRESSOR_H_

#include "../util/RandomTreeAccessor.h"

namespace FMCA {
namespace internal {

template <typename Derived>
class MPISampletMatrixCompressor {
 public:
  MPISampletMatrixCompressor() {}
  MPISampletMatrixCompressor(const SampletTreeBase<Derived> &ST, Scalar eta,
                             Scalar threshold = 0) {
    init(ST, eta, threshold);
  }

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   *
   **/
  void init(const SampletTreeBase<Derived> &ST, Scalar eta,
            Scalar threshold = 0) {
    eta_ = eta;
    threshold_ = threshold;
    rta_.init(ST, ST.block_size());
    pattern_.resize(rta_.nodes().size());

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
          if (compareCluster(pr->sons(i), *pc, eta) != LowRank)
            row_stack.push_back(std::addressof(pr->sons(i)));
        if (pc->block_id() >= pr->block_id()) {
#pragma omp critical
          pattern_[pr->block_id()].insert({pc->block_id(), Matrix(0, 0)});
        }
      }
    }
    return;
  }

  size_t pattern_structure() {
    // the column cluster tree is traversed bottom up
    const auto &rclusters = rta_.nodes();
    const auto &cclusters = rta_.nodes();
    const auto nclusters = rta_.nodes().size();
    size_t apriori_entries = 0;
    for (FMCA::Index row = 0; row < pattern_.size(); ++row) {
      const size_t map_size = pattern_[row].size();
      size_t apriori_row_entries = 0;
      for (const auto &it2 : pattern_[row]) {
        const Derived *pr = rclusters[row];
        const Derived *pc = cclusters[it2.first];
        const Index col_id = pc->block_id();
        const Index row_id = pr->block_id();
        const Index rows = pr->is_root() ? pr->Q().cols() : pr->nsamplets();
        const Index cols = pc->is_root() ? pc->Q().cols() : pc->nsamplets();
        apriori_row_entries += rows * cols;
      }
      apriori_entries += apriori_row_entries;
    }
    std::cout << "total apriori entries:         " << apriori_entries
              << std::endl;
    std::cout << "estimated memory requirements: "
              << 3. * 8. * apriori_entries / 1e9 << "GB" << std::endl;
    return apriori_entries;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen, const int my_ID = 0,
                const int num_procs = 1) {
    // the column cluster tree is traversed bottom up
    const auto &rclusters = rta_.nodes();
    const auto &cclusters = rta_.nodes();
    const auto nclusters = rta_.nodes().size();
#pragma omp parallel for schedule(dynamic)
    for (FMCA::Index row = 0; row < pattern_.size(); ++row) {
      if (row % num_procs == my_ID) {
        for (auto &it2 : pattern_[row]) {
          const Derived *pr = rclusters[row];
          const Derived *pc = cclusters[it2.first];
          const Index col_id = pc->block_id();
          const Index row_id = pr->block_id();
          Index nscalfs = 0;
          Index son_lvl = 0;
          Index offset = 0;
          size_t son_id = 0;
          Matrix &block = it2.second;
          const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());
          switch (the_case) {
              // (leaf,leaf), compute the block
              // (noleaf,leaf), compute the block
            case 3:
            case 1:
              block = recursivelyComputeBlock(*pr, *pc, e_gen);
              break;
              // (*,noleaf), recycle from right
            case 2:
            case 0:
              block.resize(pr->Q().cols(), pc->Q().rows());
              for (auto k = 0; k < pc->nSons(); ++k) {
                nscalfs = pc->sons(k).nscalfs();
                // check if pc's son is found in the row of pr
                // if so, reuse the matrix block, otherwise recompute it
                const auto it3 = pattern_[row].find(pc->sons(k).block_id());
                if (it3 != pattern_[row].end()) {
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
        }
        for (auto &it2 : pattern_[row]) {
          const Derived *pr = rclusters[row];
          const Derived *pc = cclusters[it2.first];
          Matrix &block = it2.second;
          if (!pr->is_root() && !pc->is_root())
            block = block.bottomRightCorner(pr->nsamplets(), pc->nsamplets())
                        .eval();
        }
      }
    }
    return;
  }
  /**
   *  \brief creates a posteriori thresholded triplets and stores them
   *to in the triplet list
   **/
  const std::vector<Eigen::Triplet<Scalar>> &triplets() {
    if (pattern_.size()) {
      triplet_list_.clear();
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < pattern_.size(); ++i) {
        std::vector<Triplet<Scalar>> list;
        for (auto &&it : pattern_[i]) {
          const Derived *pr = rta_.nodes()[i];
          const Derived *pc = rta_.nodes()[it.first];
          if (!pr->is_root() && !pc->is_root())
            storeBlock(
                list, pr->start_index(), pc->start_index(), pr->nsamplets(),
                pc->nsamplets(),
                it.second.bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
          else if (!pc->is_root())
            storeBlock(list, pr->start_index(), pc->start_index(),
                       pr->Q().cols(), pc->nsamplets(),
                       it.second.rightCols(pc->nsamplets()));
          else if (pr->is_root() && pc->is_root())
            storeBlock(list, pr->start_index(), pc->start_index(),
                       pr->Q().cols(), pc->Q().cols(), it.second);
          it.second.resize(0, 0);
        }
#pragma omp critical
        triplet_list_.insert(triplet_list_.end(), list.begin(), list.end());
      }
      pattern_.resize(0);
    }
    return triplet_list_;
  }

  std::vector<Eigen::Triplet<Scalar>> release_triplets() {
    std::vector<Eigen::Triplet<Scalar>> retval;
    std::swap(triplet_list_, retval);
    return retval;
  }

 private:
  /**
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  Matrix recursivelyComputeBlock(const Derived &TR, const Derived &TC,
                                 const EntryGenerator &e_gen) {
    Matrix buf(0, 0);
    // check for admissibility
    if (compareCluster(TR, TC, eta_) == LowRank) {
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
          // the row cluster is a leaf cluster: recursion on the col
          // cluster
          for (auto j = 0; j < TC.nSons(); ++j) {
            const Index nscalfs = TC.sons(j).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
            buf.conservativeResize(ret.rows(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.leftCols(nscalfs);
          }
          return buf * TC.Q();
        case 1:
          // the col cluster is a leaf cluster: recursion on the row
          // cluster
          for (auto i = 0; i < TR.nSons(); ++i) {
            const Index nscalfs = TR.sons(i).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
            buf.conservativeResize(ret.cols(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
          }
          return (buf * TR.Q()).transpose();
        case 0:
          // neither is a leaf, let recursion handle this
          for (auto i = 0; i < TR.nSons(); ++i) {
            Matrix ret1(0, 0);
            const Index r_nscalfs = TR.sons(i).nscalfs();
            for (auto j = 0; j < TC.nSons(); ++j) {
              const Index c_nscalfs = TC.sons(j).nscalfs();
              Matrix ret2 =
                  recursivelyComputeBlock(TR.sons(i), TC.sons(j), e_gen);
              ret1.conservativeResize(ret2.rows(), ret1.cols() + c_nscalfs);
              ret1.rightCols(c_nscalfs) = ret2.leftCols(c_nscalfs);
            }
            ret1 = ret1 * TC.Q();
            buf.conservativeResize(ret1.cols(), buf.cols() + r_nscalfs);
            buf.rightCols(r_nscalfs) = ret1.transpose().leftCols(r_nscalfs);
          }
          return (buf * TR.Q()).transpose();
      }
    }
    return Matrix(0, 0);
  }

  /**
   *  \brief writes a given matrix block into a-posteriori thresholded
   *         triplet format
   **/
  void storeBlock(std::vector<Eigen::Triplet<Scalar>> &triplet_buffer,
                  Index srow, Index scol, Index nrows, Index ncols,
                  const Matrix &block) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if ((srow + j <= scol + k && abs(block(j, k)) > threshold_) ||
            srow + j == scol + k)
          triplet_buffer.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
  }
  //////////////////////////////////////////////////////////////////////////////
  typedef std::map<size_t, Matrix, std::greater<size_t>> LevelBuffer;
  std::vector<Triplet<Scalar>> triplet_list_;
  std::vector<LevelBuffer> pattern_;
  RandomTreeAccessor<Derived> rta_;
  Scalar eta_;
  Scalar threshold_;
};
}  // namespace internal
}  // namespace FMCA

#endif

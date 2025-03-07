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

#include <execution>

#include "../util/RandomTreeAccessor.h"

namespace FMCA {
namespace internal {
template <typename Derived, typename ClusterComparison = CompareCluster>
class SampletMatrixCompressor {
 public:
  typedef std::map<size_t, Matrix, std::greater<size_t>> LevelBuffer;
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
          pattern_[pc->level() + pr->level()].insert({id, Matrix(0, 0)});
        }
      }
    }
    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
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
          Index i = 0;
          Index prev_i = 0;
#pragma omp atomic capture
          i = pos++;
          while (i < map_size) {
            std::advance(it2, i - prev_i);
            const Derived *pr = rclusters[it2->first % nclusters];
            const Derived *pc = cclusters[it2->first / nclusters];
            Matrix &block = it2->second;
            if (!pr->is_root() && !pc->is_root())
              block = block.bottomRightCorner(pr->nsamplets(), pc->nsamplets())
                          .eval();
            prev_i = i;
#pragma omp atomic capture
            i = pos++;
          }
        }
      }
    }
    return;
  }

  /**
   *  \brief creates a posteriori thresholded triplets and stores them to in
   *the triplet list
   **/
  std::vector<Eigen::Triplet<Scalar>> a_priori_pattern_triplets() {
    std::vector<Eigen::Triplet<Scalar>> retval;
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < pattern_.size(); ++i) {
      std::vector<Triplet<Scalar>> list;
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

  /**
   *  \brief creates a posteriori thresholded triplets and stores them to in
   *the triplet list
   **/
  const std::vector<Triplet<Scalar>> &triplets() {
    if (pattern_.size()) {
      triplet_list_.clear();
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < pattern_.size(); ++i) {
        std::vector<Triplet<Scalar>> list;
        for (auto &&it : pattern_[i]) {
          const Derived *pr = rta_.nodes()[it.first % rta_.nodes().size()];
          const Derived *pc = rta_.nodes()[it.first / rta_.nodes().size()];
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

  std::vector<Triplet<Scalar>> aposteriori_triplets_fast(const Scalar thres) {
    std::vector<Triplet<Scalar>> retval;
    std::vector<std::vector<Index>> buckets(17);
    std::vector<Scalar> norms2(17);
    constexpr Scalar invlog10 = 1. / std::log(10.);
    for (FMCA::Index i = 0; i < triplet_list_.size(); ++i) {
      const Scalar entry = std::abs(triplet_list_[i].value());
      const Scalar val = -std::floor(invlog10 * std::log(entry));
      const Index ind = val < 0 ? 0 : val;
      buckets[ind > 16 ? 16 : ind].push_back(i);
      norms2[ind > 16 ? 16 : ind] += entry * entry;
    }
    Scalar fnorm2 = 0;
    for (int i = 16; i >= 0; --i) fnorm2 += norms2[i];
    Scalar cut_snorm = 0;
    Index cut_off = 17;
    for (int i = 16; i >= 0; --i) {
      cut_snorm += norms2[i];
      if (std::sqrt(cut_snorm / fnorm2) >= thres) break;
      --cut_off;
    }
    Index ntriplets = 0;
    for (Index i = 0; i < cut_off; ++i) ntriplets += buckets[i].size();
    retval.reserve(ntriplets + npts_);
    for (Index i = 0; i < cut_off; ++i)
      for (const auto &it : buckets[i]) retval.push_back(triplet_list_[it]);
    // make sure the matrix contains the diagonal
    for (Index i = cut_off; i < 17; ++i)
      for (const auto &it : buckets[i])
        if (triplet_list_[it].row() == triplet_list_[it].col())
          retval.push_back(triplet_list_[it]);
    retval.shrink_to_fit();
    return retval;
  }

  std::vector<Triplet<Scalar>> aposteriori_triplets(const Scalar thres) {
    std::vector<Triplet<Scalar>> triplets = triplet_list_;
    if (std::abs(thres) < FMCA_ZERO_TOLERANCE) return triplets;

    // sort the triplets by magnitude, putting diagonal entries first
    // note that first sorting and then summing small to large makes
    // everything stable (positive numbers). Using Kahan summation did
    // not further improve afterwards, so we stay with fast summation
    std::vector<long int> idcs(triplet_list_.size());
    std::iota(idcs.begin(), idcs.end(), 0);
    {
      struct comp {
        comp(const std::vector<Triplet<Scalar>> &triplets) : ts_(triplets) {}
        bool operator()(const Index &a, const Index &b) const {
          const Scalar val1 = (ts_[a].row() == ts_[a].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[a].value());
          const Scalar val2 = (ts_[b].row() == ts_[b].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[b].value());
          return val1 > val2;
        }
        const std::vector<Triplet<Scalar>> &ts_;
      };
      std::stable_sort(std::execution::par_unseq, idcs.begin(), idcs.end(),
                       comp(triplet_list_));
    }

    Scalar squared_norm = 0;
    for (auto it = idcs.rbegin(); it != idcs.rend(); ++it)
      squared_norm += triplet_list_[*it].value() * triplet_list_[*it].value();

    Scalar cut_snorm = 0;
    Index cut_off = triplet_list_.size();
    for (auto it = idcs.rbegin(); it != idcs.rend(); ++it) {
      cut_snorm += triplet_list_[*it].value() * triplet_list_[*it].value();
      if (std::sqrt(cut_snorm / squared_norm) >= thres) break;
      --cut_off;
    }
    // keep at least the diagonal
    cut_off = cut_off < npts_ ? npts_ : cut_off;
    idcs.resize(cut_off);
    triplets.resize(cut_off);
    for (Index i = 0; i < cut_off; ++i) triplets[i] = triplet_list_[idcs[i]];
    return triplets;
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
    if (ClusterComparison::compare(TR, TC, eta_) == LowRank) {
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
          for (auto j = 0; j < TC.nSons(); ++j) {
            const Index nscalfs = TC.sons(j).nscalfs();
            Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
            buf.conservativeResize(ret.rows(), buf.cols() + nscalfs);
            buf.rightCols(nscalfs) = ret.leftCols(nscalfs);
          }
          return buf * TC.Q();
        case 1:
          // the col cluster is a leaf cluster: recursion on the row cluster
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
        if ((srow + j <= scol + k && std::abs(block(j, k)) > threshold_) ||
            (srow == scol && j == k))
          triplet_buffer.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
  }

  /**
   *  \brief writes a given matrix block into a-posteriori thresholded
   *         triplet format
   **/
  void storeBlock2(std::vector<Eigen::Triplet<Scalar>> &triplet_buffer,
                   Index srow, Index scol, Index nrows, Index ncols,
                   const Matrix &block) {
    struct comp {
      bool operator()(const Eigen::Triplet<Scalar> &a,
                      const Eigen::Triplet<Scalar> &b) const {
        const Scalar val1 =
            (a.row() == a.col()) ? FMCA_INF : std::abs(a.value());
        const Scalar val2 =
            (b.row() == b.col()) ? FMCA_INF : std::abs(b.value());
        return val1 > val2;
      }
    };
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(block.size());
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if ((srow + j <= scol + k &&
             std::abs(block(j, k)) > FMCA_ZERO_TOLERANCE) ||
            (srow == scol && j == k))
          triplets.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
    std::sort(triplets.begin(), triplets.end(), comp());

    Scalar squared_norm = 0;
    for (auto it = triplets.rbegin(); it != triplets.rend(); ++it)
      squared_norm += it->value() * it->value();
    Scalar cut_snorm = 0;
    Index cut_off = triplets.size();
    for (auto it = triplets.rbegin(); it != triplets.rend(); ++it) {
      cut_snorm += it->value() * it->value();
      if (std::sqrt(cut_snorm / squared_norm) >= threshold_) break;
      --cut_off;
    }
    // keep at least the diagonal
    if (srow == scol) cut_off = cut_off < nrows ? nrows : cut_off;
    triplets.resize(cut_off);
    triplet_buffer.insert(triplet_buffer.end(), triplets.begin(),
                          triplets.end());
  }

  void storeEmptyBlock(std::vector<Eigen::Triplet<Scalar>> &triplet_buffer,
                       Index srow, Index scol, Index nrows, Index ncols) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (srow + j <= scol + k)
          triplet_buffer.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, 0));
  }
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Triplet<Scalar>> triplet_list_;
  std::vector<LevelBuffer> pattern_;
  RandomTreeAccessor<Derived> rta_;
  Scalar eta_;
  Scalar threshold_;
  Index npts_;
};
}  // namespace internal
}  // namespace FMCA

#endif

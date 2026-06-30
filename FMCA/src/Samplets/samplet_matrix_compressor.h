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

#include "../util/MemoryArena.h"
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
    max_size_ = 0;
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
            pattern_[pc->level() + pr->level()].insert({id, Matrix(0, 0)});
            max_size_ = std::max<std::ptrdiff_t>(
                {max_size_, pr->Q().rows(), pr->Q().cols(), pr->V().cols()});
            max_size_ = std::max<std::ptrdiff_t>(
                {max_size_, pc->Q().rows(), pc->Q().cols(), pc->V().cols()});
          }
        }
      }
    }
    std::cout << "determined maximum mem size:  " << max_size_ << std::endl;
    mem_arena_.init(3 * max_size_ * max_size_);
    return;
  }

  template <typename EntGenerator>
  void compress(const EntGenerator &e_gen) {
    // the column cluster tree is traversed bottom up
    const auto &rclusters = rta_.nodes();
    const auto &cclusters = rta_.nodes();
    const auto nclusters = rta_.nodes().size();
    Index callct[5] = {0, 0, 0, 0, 0};
    Index maxrec = 0;
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
          Matrix buf;
          const char the_case = 2 * (!pr->nSons()) + (!pc->nSons());
          switch (the_case) {
            // (leaf,leaf), compute the block
            case 3: {
              std::unique_ptr<Scalar[]> mem = mem_arena_.acquire();
              recursivelyComputeBlock_noalloc(*pr, *pc, e_gen, mem.get(),
                                              max_size_ * max_size_, callct);
              maxrec = std::max(maxrec, callct[4]);
              callct[4] = 0;
              assert(pr->Q().cols() <= max_size_ &&
                     pc->Q().cols() <= max_size_ &&
                     "mem mismatch at comp case 3");
              Map<Matrix> retval(mem.get(), pr->Q().cols(), pc->Q().cols());
              mem_arena_.release(std::move(mem));
              block = retval;
              break;
            }
            // (noleaf,leaf), recycle from below
            case 1: {
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
                  std::unique_ptr<Scalar[]> mem = mem_arena_.acquire();
                  recursivelyComputeBlock_noalloc(
                      pr->sons(k), *pc, e_gen, mem.get(), max_size_ * max_size_,
                      callct);
                  maxrec = std::max(maxrec, callct[4]);
                  callct[4] = 0;
                  assert(pr->sons(k).Q().cols() <= max_size_ &&
                         pc->Q().cols() <= max_size_ &&
                         "mem mismatch at comp case 1");
                  Map<Matrix> ret(mem.get(), pr->sons(k).Q().cols(),
                                  pc->Q().cols());
                  block.middleRows(offset, nscalfs) = ret.topRows(nscalfs);
                  mem_arena_.release(std::move(mem));
                }
                offset += nscalfs;
              }
              block = pr->Q().transpose() * block;
              break;
            }
              // (*,noleaf), recycle from right
            case 2:
            case 0: {
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
                  std::unique_ptr<Scalar[]> mem = mem_arena_.acquire();
                  recursivelyComputeBlock_noalloc(
                      *pr, pc->sons(k), e_gen, mem.get(), max_size_ * max_size_,
                      callct);
                  maxrec = std::max(maxrec, callct[4]);
                  callct[4] = 0;
                  assert(pr->Q().cols() <= max_size_ &&
                         pc->sons(k).Q().cols() <= max_size_ &&
                         "mem mismatch at comp case 0/2");
                  Map<Matrix> ret(mem.get(), pr->Q().cols(),
                                  pc->sons(k).Q().cols());
                  block.middleCols(offset, nscalfs) = ret.leftCols(nscalfs);
                  mem_arena_.release(std::move(mem));
                }
                offset += nscalfs;
              }
              block = block * pc->Q();
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
    std::cout << "final arena size: " << mem_arena_.slabs_in_use() << "/"
              << mem_arena_.num_free_slabs() << std::endl;
    std::cout << "case count: " << callct[0] << "," << callct[1] << ","
              << callct[2] << "," << callct[3] << " max depth: " << maxrec
              << std::endl;

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

  /**
   *  \brief creates a posteriori thresholded triplets and stores them to in
   *the triplet list
   **/
  const std::vector<Triplet> &triplets() {
    if (pattern_.size()) {
      triplet_list_.clear();
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < pattern_.size(); ++i) {
        std::vector<Triplet> list;
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

  std::vector<Triplet> aposteriori_triplets_fast(const Scalar thres) {
    std::vector<Triplet> retval;
    std::vector<std::vector<Index>> buckets(17);
    std::vector<Scalar> norms2(17);
    const Scalar invlog10 = 1. / std::log(10.);
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

  std::vector<Triplet> aposteriori_triplets(const Scalar thres) {
    std::vector<Triplet> triplets = triplet_list_;
    if (std::abs(thres) < FMCA_ZERO_TOLERANCE) return triplets;

    // sort the triplets by magnitude, putting diagonal entries first
    // note that first sorting and then summing small to large makes
    // everything stable (positive numbers). Using Kahan summation did
    // not further improve afterwards, so we stay with fast summation
    std::vector<long int> idcs(triplet_list_.size());
    std::iota(idcs.begin(), idcs.end(), 0);
    {
      struct comp {
        comp(const std::vector<Triplet> &triplets) : ts_(triplets) {}
        bool operator()(const Index &a, const Index &b) const {
          const Scalar val1 = (ts_[a].row() == ts_[a].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[a].value());
          const Scalar val2 = (ts_[b].row() == ts_[b].col())
                                  ? FMCA_INF
                                  : std::abs(ts_[b].value());
          return val1 > val2;
        }
        const std::vector<Triplet> &ts_;
      };
      std::sort(idcs.begin(), idcs.end(), comp(triplet_list_));
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

  std::vector<Triplet> release_triplets() {
    std::vector<Triplet> retval;
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
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  void recursivelyComputeBlock_noalloc(const Derived &TR, const Derived &TC,
                                       const EntryGenerator &e_gen, Scalar *mem,
                                       Index stride, Index *callct) {
    // check for admissibility
    if (ClusterComparison::compare(TR, TC, eta_) == LowRank) {
      e_gen.interpolate_kernel_noalloc(TR, TC, mem, stride);
      assert(stride >= TR.V().rows() * TC.V().rows());
      Map<Matrix> buf(mem + 2 * stride, TR.V().rows(), TC.V().rows());
      assert(stride >= TR.V().rows() * TC.V().cols());
      Map<Matrix> temp(mem + stride, TR.V().rows(), TC.V().cols());
      temp.noalias() = buf * TC.V();
      assert(stride >= TR.Q().cols() * TC.Q().cols());
      Map<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
      retval.noalias() = TR.V().transpose() * temp;
      return;
    } else {
      const char the_case = 2 * (!TR.nSons()) + !TC.nSons();
      ++(callct[the_case]);
      switch (the_case) {
        case 3: {
          // both are leafs: compute the block and return
          e_gen.compute_dense_block_noalloc(TR, TC, mem, stride);
          assert(stride >= TR.Q().rows() * TC.Q().rows());
          Map<Matrix> buf(mem + 2 * stride, TR.Q().rows(), TC.Q().rows());
          assert(stride >= TR.Q().rows() * TC.Q().cols());
          Map<Matrix> temp(mem + stride, TR.Q().rows(), TC.Q().cols());
          temp.noalias() = buf * TC.Q();
          assert(stride >= TR.Q().cols() * TC.Q().cols());
          Map<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = TR.Q().transpose() * temp;
          return;
        }
        case 2: {
          Map<Matrix> buf(mem + 2 * stride, TR.Q().cols(), TC.Q().rows());
          // the row cluster is a leaf cluster: recursion on the col cluster
          Index offset = 0;
          ++(callct[4]);

          for (auto j = 0; j < TC.nSons(); ++j) {
            std::unique_ptr<Scalar[]> r_mem = mem_arena_.acquire();
            recursivelyComputeBlock_noalloc(TR, TC.sons(j), e_gen, r_mem.get(),
                                            stride, callct);
            Map<Matrix> temp(r_mem.get(), TR.Q().cols(), TC.sons(j).Q().cols());
            const Index nscalfs = TC.sons(j).nscalfs();
            buf.middleCols(offset, nscalfs) = temp.leftCols(nscalfs);
            offset += nscalfs;
            mem_arena_.release(std::move(r_mem));
          }
          Map<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = buf * TC.Q();
          return;
        }
        case 1: {
          Map<Matrix> buf(mem + 2 * stride, TR.Q().rows(), TC.Q().cols());
          // the col cluster is a leaf cluster: recursion on the row cluster
          Index offset = 0;
          ++(callct[4]);

          for (auto i = 0; i < TR.nSons(); ++i) {
            std::unique_ptr<Scalar[]> c_mem = mem_arena_.acquire();
            recursivelyComputeBlock_noalloc(TR.sons(i), TC, e_gen, c_mem.get(),
                                            stride, callct);
            Map<Matrix> temp(c_mem.get(), TR.sons(i).Q().cols(), TC.Q().cols());
            const Index nscalfs = TR.sons(i).nscalfs();
            buf.middleRows(offset, nscalfs) = temp.topRows(nscalfs);
            offset += nscalfs;
            mem_arena_.release(std::move(c_mem));
          }
          Map<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = TR.Q().transpose() * buf;
          return;
        }
        case 0: {
          Map<Matrix> buf(mem + 2 * stride, TR.Q().rows(), TC.Q().cols());
          // neither is a leaf, let recursion handle this
          Index r_offset = 0;
          ++(callct[4]);

          for (auto i = 0; i < TR.nSons(); ++i) {
            Map<Matrix> cbuf(mem, TR.sons(i).Q().cols(), TC.Q().rows());
            Index c_offset = 0;
            for (auto j = 0; j < TC.nSons(); ++j) {
              std::unique_ptr<Scalar[]> r_mem = mem_arena_.acquire();
              recursivelyComputeBlock_noalloc(TR.sons(i), TC.sons(j), e_gen,
                                              r_mem.get(), stride, callct);
              Map<Matrix> temp(r_mem.get(), TR.sons(i).Q().cols(),
                               TC.sons(j).Q().cols());
              const Index c_nscalfs = TC.sons(j).nscalfs();
              cbuf.middleCols(c_offset, c_nscalfs) = temp.leftCols(c_nscalfs);
              c_offset += c_nscalfs;
              mem_arena_.release(std::move(r_mem));
            }
            Map<Matrix> res_buf(mem + stride, TR.sons(i).Q().cols(),
                                TC.Q().cols());
            res_buf.noalias() = cbuf * TC.Q();
            const Index r_nscalfs = TR.sons(i).nscalfs();
            buf.middleRows(r_offset, r_nscalfs) = res_buf.topRows(r_nscalfs);
            r_offset += r_nscalfs;
          }
          Map<Matrix> retval(mem, TR.Q().cols(), TC.Q().cols());
          retval.noalias() = TR.Q().transpose() * buf;
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
  void storeBlock(std::vector<Triplet> &triplet_buffer, Index srow, Index scol,
                  Index nrows, Index ncols, const Matrix &block) {
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
  MemoryArena<Scalar> mem_arena_;
  std::vector<Triplet> triplet_list_;
  std::vector<LevelBuffer> pattern_;
  RandomTreeAccessor<Derived> rta_;
  std::ptrdiff_t max_size_;
  Scalar eta_;
  Scalar threshold_;
  Index npts_;
};  // namespace internal
}  // namespace internal
}  // namespace FMCA

#endif

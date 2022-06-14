// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_OMPSAMPLETCOMPRESSOR_H_
#define FMCA_SAMPLETS_OMPSAMPLETCOMPRESSOR_H_

namespace FMCA {

template <typename Derived>
class ompSampletCompressor {
 public:
  ompSampletCompressor() {}
  ompSampletCompressor(const SampletTreeBase<Derived> &ST, Scalar eta) {
    init(ST, eta);
  }

  /**
   *  \brief creates the matrix pattern based on the cluster tree and the
   *         admissibility condition
   *
   **/
  void init(const SampletTreeBase<Derived> &ST, Scalar eta,
            Scalar threshold = 1e-6) {
    eta_ = eta;
    threshold_ = threshold;
    n_clusters_ = std::distance(ST.cbegin(), ST.cend());
    if (const char *env_p = std::getenv("OMP_NUM_THREADS"))
      std::cout << "OMP_NUM_THREADS:          " << env_p << std::endl;
    s_mapper_.resize(n_clusters_);
    m_blx_.resize(n_clusters_, n_clusters_);
    max_level_ = 0;
    {
      // first sweep to determine samplet tree characteristics
      std::vector<const Derived *> row_stack;
      std::vector<const Derived *> col_stack;
      row_stack.push_back(std::addressof(ST.derived()));
      while (row_stack.size()) {
        const Derived *pr = row_stack.back();
        row_stack.pop_back();
        s_mapper_[pr->block_id()] = pr;
        max_level_ = max_level_ < pr->level() ? pr->level() : max_level_;
        for (auto i = 0; i < pr->nSons(); ++i)
          row_stack.push_back(std::addressof(pr->sons(i)));
        assert(col_stack.size() == 0 && "col: non-empty stack at return");
        col_stack.push_back(std::addressof(ST.derived()));
        while (col_stack.size()) {
          const Derived *pc = col_stack.back();
          col_stack.pop_back();
          for (auto i = 0; i < pc->nSons(); ++i) {
            if (compareCluster(pc->sons(i), *pr, eta) != LowRank)
              col_stack.push_back(std::addressof(pc->sons(i)));
          }
          // we create here the lower triangular pattern, which is later on
          // used in the transposed way to obtain fast column access for
          // the matrix assembly
          if (pr->block_id() >= pc->block_id())
            m_blx_(pr->block_id(), pc->block_id()).resize(0, 0);
        }
      }
      assert(row_stack.size() == 0 && "row: non-empty stack at return");
    }
    return;
  }

  template <typename EntGenerator>
  void compress(const SampletTreeBase<Derived> &ST, const EntGenerator &e_gen) {
    m_blx_.resize(n_clusters_, n_clusters_);
    for (auto j = 1; j <= n_clusters_; ++j) {
      const Derived *pc = s_mapper_[n_clusters_ - j];
      const Index col_id = pc->block_id();
      setup_level_mapper_(n_clusters_ - j);
      for (auto it = lvl_mapper_.rbegin(); it != lvl_mapper_.rend(); ++it) {
#pragma omp parallel for
        for (int i = 0; i < it->size(); ++i) {
          const Derived *pr = (*it)[i];
          const Index row_id = pr->block_id();
          const Index pos1 = m_blx_.find(row_id, col_id);
          const Index pos2 = m_blx_.find(col_id, row_id);
          if (pos2 < m_blx_.idx()[col_id].size()); else assert(false && "wtf");
          Matrix &block = m_blx_(row_id, col_id);
          block.resize(0, 0);
          if (pr->nSons() && row_id < col_id) {
            for (auto k = 0; k < pr->nSons(); ++k) {
              const Index nscalfs = pr->sons(k).nscalfs();
              const Index son_id = pr->sons(k).block_id();
              // check if pc is found in the row of pr's son
              const Index pos = m_blx_.find(son_id, col_id);
              // if so, reuse the matrix block, otherwise recompute it
              if (pos < m_blx_.idx()[son_id].size()) {
                Matrix &ret = m_blx_.val()[son_id][pos];
                block.conservativeResize(ret.cols(), block.cols() + nscalfs);
                block.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
              } else {
                const Matrix ret =
                    recursivelyComputeBlock(pr->sons(k), *pc, e_gen);
                block.conservativeResize(ret.cols(), block.cols() + nscalfs);
                block.rightCols(nscalfs) = ret.transpose().leftCols(nscalfs);
              }
            }
            block = (block * pr->Q()).transpose();
#ifdef FMCA_DEBUG_FLAG_
            Matrix ret = recursivelyComputeBlock(*pr, *pc, e_gen);
            assert(block.rows() == pr->Q().cols() && "row mismatch 1");
            assert(block.cols() == pc->Q().cols() && "col mismatch 1");
            assert((ret - block).norm() / ret.norm() < 1e-4 && "wtf1");
#endif
          } else {
            if (!pc->nSons())
              block = recursivelyComputeBlock(*pr, *pc, e_gen);
            else {
              for (auto k = 0; k < pc->nSons(); ++k) {
                const Index nscalfs = pc->sons(k).nscalfs();
                const Index son_id = pc->sons(k).block_id();
                // check if pc's son is found in the row of pr
                const Index pos = m_blx_.find(row_id, son_id);
                // if so, reuse the matrix block, otherwise recompute it
                if (pos < m_blx_.idx()[row_id].size()) {
                  const Matrix &ret = m_blx_.val()[row_id][pos];
                  block.conservativeResize(ret.rows(), block.cols() + nscalfs);
                  block.rightCols(nscalfs) = ret.leftCols(nscalfs);
                } else {
                  const Matrix ret =
                      recursivelyComputeBlock(*pr, pc->sons(k), e_gen);
                  block.conservativeResize(ret.rows(), block.cols() + nscalfs);
                  block.rightCols(nscalfs) = ret.leftCols(nscalfs);
                }
              }
              block = block * pc->Q();
#ifdef FMCA_DEBUG_FLAG_
              Matrix ret = recursivelyComputeBlock(*pr, *pc, e_gen);
              assert(block.rows() == pr->Q().cols() && "row mismatch 2");
              assert(block.cols() == pc->Q().cols() && "col mismatch 2");
              assert((ret - block).norm() / ret.norm() < 1e-4 && "wtf2");
#endif
            }
          }
        }
      }
    }
    std::cout << "compute_calls: " << compute_calls_ << " percent low rank: "
              << Scalar(lowrank_calls_) / compute_calls_ * 100 << std::endl;
    return;
  }

  /**
   *  \brief creates a posteriori thresholded triplets and stores them to in
   *the triplet list
   **/
  const std::vector<Eigen::Triplet<Scalar>> &triplets() {
    triplet_list_.clear();

    for (int i = n_clusters_ - 1; i >= 0; --i) {
      const auto &idx = m_blx_.idx()[i];
      const std::vector<Matrix> &val = m_blx_.val()[i];
      const Derived *pr = s_mapper_[i];
      for (int j = 0; j < idx.size(); ++j) {
        const Derived *pc = s_mapper_[idx[j]];
        if (!pr->is_root() && !pc->is_root())
          storeBlock(
              pr->start_index(), pc->start_index(), pr->nsamplets(),
              pc->nsamplets(),
              val[j].bottomRightCorner(pr->nsamplets(), pc->nsamplets()));
        else if (!pc->is_root())
          storeBlock(pr->start_index(), pc->start_index(), pr->Q().cols(),
                     pc->nsamplets(), val[j].rightCols(pc->nsamplets()));
        else if (pr->is_root() && pc->is_root())
          storeBlock(pr->start_index(), pc->start_index(), pr->Q().cols(),
                     pc->Q().cols(), val[j]);
      }
    }
    return triplet_list_;
  }

 private:
  /**
   *  \brief maps a given block matrix column into a levelwise synchronized
   *         container
   **/
  void setup_level_mapper_(Index i) {
    lvl_mapper_.clear();
    lvl_mapper_.resize(max_level_ + 1);
    for (auto &&it : m_blx_.idx()[i]) {
      const Derived *cluster = s_mapper_[it];
      lvl_mapper_[cluster->level()].push_back(cluster);
    }
  }

  /**
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  Matrix recursivelyComputeBlock(const Derived &TR, const Derived &TC,
                                 const EntryGenerator &e_gen) {
    ++compute_calls_;
    Matrix buf(0, 0);
    Matrix retval(0, 0);
    // check for admissibility
    if (compareCluster(TR, TC, eta_) == LowRank) {
      ++lowrank_calls_;
      e_gen.interpolate_kernel(TR, TC, &buf);
      retval = TR.V().transpose() * buf * TC.V();
    } else {
      if (!TR.nSons() && !TC.nSons()) {
        // both are leafs: compute the block and return
        e_gen.compute_dense_block(TR, TC, &buf);
        retval = TR.Q().transpose() * buf * TC.Q();
      } else if (!TR.nSons() && TC.nSons()) {
        // the row cluster is a leaf cluster: recursion on the col cluster
        for (auto j = 0; j < TC.nSons(); ++j) {
          Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
          buf.conservativeResize(ret.rows(), buf.cols() + TC.sons(j).nscalfs());
          buf.rightCols(TC.sons(j).nscalfs()) =
              ret.leftCols(TC.sons(j).nscalfs());
        }
        retval = buf * TC.Q();
      } else if (TR.nSons() && !TC.nSons()) {
        // the col cluster is a leaf cluster: recursion on the row cluster
        for (auto i = 0; i < TR.nSons(); ++i) {
          Matrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
          buf.conservativeResize(ret.cols(), buf.cols() + TR.sons(i).nscalfs());
          buf.rightCols(TR.sons(i).nscalfs()) =
              ret.transpose().leftCols(TR.sons(i).nscalfs());
        }
        retval = (buf * TR.Q()).transpose();
      } else {
        // neither is a leaf, let recursion handle this
        for (auto i = 0; i < TR.nSons(); ++i) {
          Matrix ret1(0, 0);
          for (auto j = 0; j < TC.nSons(); ++j) {
            Matrix ret2 =
                recursivelyComputeBlock(TR.sons(i), TC.sons(j), e_gen);
            ret1.conservativeResize(ret2.rows(),
                                    ret1.cols() + TC.sons(j).nscalfs());
            ret1.rightCols(TC.sons(j).nscalfs()) =
                ret2.leftCols(TC.sons(j).nscalfs());
          }
          ret1 = ret1 * TC.Q();
          buf.conservativeResize(ret1.cols(),
                                 buf.cols() + TR.sons(i).nscalfs());
          buf.rightCols(TR.sons(i).nscalfs()) =
              ret1.transpose().leftCols(TR.sons(i).nscalfs());
        }
        retval = (buf * TR.Q()).transpose();
      }
    }
    return retval;
  }

  /**
   *  \brief writes a given matrix block into a-posteriori thresholded
   *         triplet format
   **/
  template <typename otherDerived>
  void storeBlock(Index srow, Index scol, Index nrows, Index ncols,
                  const Eigen::MatrixBase<otherDerived> &block) {
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (srow + j <= scol + k && abs(block(j, k)) > threshold_)
          triplet_list_.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
  }

  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<Scalar>> triplet_list_;
  FMCA::SparseMatrix<Matrix> m_blx_;
  std::vector<const Derived *> s_mapper_;
  std::vector<std::vector<const Derived *>> lvl_mapper_;
  Index n_clusters_;
  Index max_level_;
  Index compute_calls_;
  Index lowrank_calls_;
  Scalar eta_;
  Scalar threshold_;
};
}  // namespace FMCA

#endif

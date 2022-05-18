// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_SAMPLETS_BIVARIATECOMPRESSORBASE_H_
#define FMCA_SAMPLETS_BIVARIATECOMPRESSORBASE_H_
//#define FMCA_COMPRESSOR_BUFSIZE_
//#define FMCA_SYMMETRIC_STORAGE_

#include "../util/SparseMatrix.h"

namespace FMCA {

template <typename Derived> struct unsymmetric_compressor_impl {

  template <typename EntryGenerator>
  void compress(SampletTreeBase<Derived> &TR, SampletTreeBase<Derived> &TC,
                const EntryGenerator &e_gen, Scalar eta = 0.8,
                Scalar threshold = 1e-6) {
    eigen_assert(TR.is_root() && TC.is_root() &&
                 "compress needs to be called from root");
    eta_ = eta;
    threshold_ = threshold;
    triplet_list_.clear();
    buffer_.clear();
    const Index row_samplet_blocks = std::distance(TR.cbegin(), TR.cend());
    const Index col_samplet_blocks = std::distance(TC.cbegin(), TC.cend());
    buffer_.resize(row_samplet_blocks);
    max_buff_size_ = 0;
    storage_size_ = 0;
    ////////////////////////////////////////////////////////////////////////////
    // set up the compressed matrix
    PB_.reset(col_samplet_blocks);

    setupColumn(TR.derived(), TC.derived(), e_gen);
    // set up remainder of the first column
    for (const auto &cluster : TR) {
      if (!cluster.is_root()) {
        const Index block_id = cluster.derived().block_id();
        const Index start_index = cluster.derived().start_index();
        const Index nsamplets = cluster.derived().nsamplets();
        const Index nscalfs = cluster.derived().nscalfs();
        auto it = buffer_[block_id].find(TC.derived().block_id());
        eigen_assert(it != buffer_[block_id].end() &&
                     "there is a missing root block!");
        storeBlock(start_index, TR.derived().start_index(), nsamplets,
                   TC.derived().Q().cols(), (it->second).bottomRows(nsamplets));
        buffer_[block_id].erase(it);
      }
    }
#ifdef FMCA_COMPRESSOR_BUFSIZE_
    std::cout << "max buffer size: " << max_buff_size_ << std::endl;
    max_buff_size_ = 0;
    for (const auto &it : buffer_)
      max_buff_size_ += it.size();
    std::cout << "final buffer size: " << max_buff_size_ << std::endl;
#endif
  }

  //////////////////////////////////////////////////////////////////////////////
  const std::vector<Eigen::Triplet<Scalar>> &pattern_triplets() const {
    return triplet_list_;
  }
  Matrix matrix() const {
    Eigen::SparseMatrix<Scalar> S(N_, N_);
    S.setFromTriplets(triplet_list_.begin(), triplet_list_.end());
    return Matrix(S);
  }

  SparseMatrix<Scalar> pattern(SampletTreeBase<Derived> &ST,
                                   Scalar eta = 0.8) {
    SparseMatrix<Scalar> retval(ST.indices().size(), ST.indices().size());
    for (const auto &TC : ST) {
      const Index c_start = TC.start_index();
      for (const auto &TR : ST) {
        const Index r_start = TR.start_index();
        // is there an entry?
        if (compareCluster(TR, TC) != LowRank) {
          if (TC.is_root()) {
            // set up first column
            if (TR.is_root())
              for (auto j = c_start; j < c_start + TC.Q().cols(); ++j)
                for (auto i = r_start; i < r_start + TR.Q().cols(); ++i)
                  retval(i, j) = 1;
            else if (TR.nsamplets())
              for (auto j = c_start; j < c_start + TC.Q().cols(); ++j)
                for (auto i = r_start; i < r_start + TR.nsamplets(); ++i)
                  retval(i, j) = 1;
          } else if (TC.nsamplets()) {
            // set up remainder of the first row
            if (TR.is_root())
              for (auto j = c_start; j < c_start + TC.nsamplets(); ++j)
                for (auto i = r_start; i < r_start + TR.Q().cols(); ++i)
                  retval(i, j) = 1;
            else if (TR.nsamplets())
              for (auto j = c_start; j < c_start + TC.nsamplets(); ++j)
                for (auto i = r_start; i < r_start + TR.nsamplets(); ++i)
                  retval(i, j) = 1;
          }
        }
      }
    }
    return retval;
  }
  /**
   *  \brief recursively computes for a given pair of row and column
   *clusters the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi,
   *A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  Matrix recursivelyComputeBlock(const Derived &TR, const Derived &TC,
                                      const EntryGenerator &e_gen) {
    Matrix buf(0, 0);
    Matrix retval(0, 0);
    // check for admissibility
    if (compareCluster(TR, TC, eta_) == LowRank) {
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
  //////////////////////////////////////////////////////////////////////////////
  template <typename EntryGenerator>
  void setupRow(const Derived &TR, const Derived &TC,
                const EntryGenerator &e_gen) {
    Matrix block(0, 0);
    Matrix buf(0, 0);
    ////////////////////////////////////////////////////////////////////////////
    // if there are children of the row cluster, we proceed recursively
    if (TR.nSons()) {
      for (auto i = 0; i < TR.nSons(); ++i) {
        if (compareCluster(TR.sons(i), TC, eta_) != LowRank) {
          setupRow(TR.sons(i), TC, e_gen);
          auto it = buffer_[TR.sons(i).block_id()].find(TC.block_id());
          eigen_assert(it != buffer_[TR.sons(i).block_id()].end() &&
                       "row: entry does not exist");
          buf.conservativeResize((it->second).cols(),
                                 buf.cols() + TR.sons(i).nscalfs());
          buf.rightCols(TR.sons(i).nscalfs()) =
              (it->second).transpose().leftCols(TR.sons(i).nscalfs());
          if (it->first != 0)
            buffer_[TR.sons(i).block_id()].erase(it);
        } else {
          Matrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
          buf.conservativeResize(ret.cols(), buf.cols() + TR.sons(i).nscalfs());
          buf.rightCols(TR.sons(i).nscalfs()) =
              ret.transpose().leftCols(TR.sons(i).nscalfs());
        }
      }
      block = (buf * TR.Q()).transpose();
      // we are at a leaf of the row cluster tree
    } else {
      // if TR and TC are both leafs, we compute the corresponding matrix
      // block
      if (!TC.nSons())
        block = recursivelyComputeBlock(TR, TC, e_gen);
      // if TC is not a leaf, we reuse the blocks of its children
      else {
        for (auto j = 0; j < TC.nSons(); ++j) {
          auto it = buffer_[TR.block_id()].find(TC.sons(j).block_id());
          if (it != buffer_[TR.block_id()].end()) {
            buf.conservativeResize((it->second).rows(),
                                   buf.cols() + TC.sons(j).nscalfs());
            buf.rightCols(TC.sons(j).nscalfs()) =
                (it->second).leftCols(TC.sons(j).nscalfs());
          } else {
            Matrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
            buf.conservativeResize(ret.rows(),
                                   buf.cols() + TC.sons(j).nscalfs());
            buf.rightCols(TC.sons(j).nscalfs()) =
                ret.leftCols(TC.sons(j).nscalfs());
          }
        }
        block = buf * TC.Q();
      }
      for (auto j = 0; j < TC.nSons(); ++j)
        buffer_[TR.block_id()].erase(TC.sons(j).block_id());
    }
    if (TR.nsamplets() && TC.nsamplets() && !TC.is_root() && !TR.is_root())
      storeBlock(TR.start_index(), TC.start_index(), TR.nsamplets(),
                 TC.nsamplets(),
                 block.bottomRightCorner(TR.nsamplets(), TC.nsamplets()));
    buffer_[TR.block_id()].emplace(std::make_pair(TC.block_id(), block));
#ifdef FMCA_COMPRESSOR_BUFSIZE_
    Index buff_size = 0;
    for (const auto &it : buffer_)
      buff_size += it.size();
    max_buff_size_ = max_buff_size_ < buff_size ? buff_size : max_buff_size_;
    return;
#endif
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename EntryGenerator>
  void setupColumn(const Derived &TR, const Derived &TC,
                   const EntryGenerator &e_gen) {
    Matrix retval;
    if (TC.nSons())
      for (auto i = 0; i < TC.nSons(); ++i)
        setupColumn(TR, TC.sons(i), e_gen);
    setupRow(TR, TC, e_gen);
    auto it = buffer_[TR.block_id()].find(TC.block_id());
    eigen_assert(it != buffer_[TR.block_id()].end() &&
                 "col: there is a missing root block!");
    if (TC.is_root())
      storeBlock(TR.start_index(), TC.start_index(), TR.Q().cols(),
                 TC.Q().cols(), it->second);
    // otherwise store  [A^PhiSigma; A^SigmaSigma]
    else if (TC.nsamplets())
      storeBlock(TR.start_index(), TC.start_index(), TR.Q().cols(),
                 TC.nsamplets(), (it->second).rightCols(TC.nsamplets()));
    buffer_[TR.block_id()].erase(it);
    PB_.next();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  void storeBlock(Index srow, Index scol, Index nrows,
                  Index ncols,
                  const Eigen::MatrixBase<otherDerived> &block) {
    storage_size_ += ncols * nrows;
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (abs(block(j, k)) > threshold_)
#ifdef FMCA_SYMMETRIC_STORAGE_
          if (srow + j >= scol + k)
            triplet_list_.push_back(
                Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
#else
          triplet_list_.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
#endif
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<Scalar>> triplet_list_;
  std::vector<std::map<Index, Matrix>> buffer_;
  ProgressBar PB_;
  Scalar eta_;
  Scalar threshold_;
  Index N_;
  size_t storage_size_;
  size_t max_buff_size_;
};
} // namespace FMCA
#endif

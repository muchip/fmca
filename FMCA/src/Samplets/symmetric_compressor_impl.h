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
#ifndef FMCA_SAMPLETS_SYMMETRICCOMPRESSORIMPL_H_
#define FMCA_SAMPLETS_SYMMETRICCOMPRESSORIMPL_H_
//#define FMCA_COMPRESSOR_BUFSIZE_

namespace FMCA {

template <typename Derived> struct symmetric_compressor_impl {
  enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;

  template <typename EntryGenerator>
  void compress(const SampletTreeBase<Derived> &ST, const EntryGenerator &e_gen,
                value_type eta = 0.8, value_type threshold = 1e-6) {
    eigen_assert(ST.is_root() && "compress needs to be called from root");
    eta_ = eta;
    threshold_ = threshold;
    triplet_list_.clear();
    triplet_list_.reserve(1230 * ST.derived().indices().size());
    std::cout << "trips reserved\n" << std::flush;
    buffer_.clear();
    const IndexType n_samplet_blocks = std::distance(ST.cbegin(), ST.cend());
    buffer_.resize(n_samplet_blocks);
    max_buff_size_ = 0;
    storage_size_ = 0;
    ////////////////////////////////////////////////////////////////////////////
    // set up the compressed matrix
    PB_.reset(n_samplet_blocks);
    compute_block_calls_ = 0;

    setupColumn(ST.derived(), ST.derived(), e_gen);
    triplet_list_.shrink_to_fit();
    std::cout << std::endl;
#ifdef FMCA_COMPRESSOR_BUFSIZE_
    std::cout << "compute calls: " << compute_block_calls_ << std::endl;
    std::cout << "max buffer size: " << max_buff_size_ << std::endl;
    max_buff_size_ = 0;
    for (const auto &it : buffer_)
      max_buff_size_ += it.size();
    std::cout << "final buffer size: " << max_buff_size_ << std::endl;
#endif
  }

  //////////////////////////////////////////////////////////////////////////////
  value_type computeDistance(const Derived &TR, const Derived &TC) {
    const value_type row_radius = 0.5 * TR.bb().col(2).norm();
    const value_type col_radius = 0.5 * TC.bb().col(2).norm();
    const value_type dist = 0.5 * (TR.bb().col(0) - TC.bb().col(0) +
                                   TR.bb().col(1) - TC.bb().col(1))
                                      .norm() -
                            row_radius - col_radius;
    return dist > 0 ? dist : 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  Admissibility compareCluster(const Derived &cluster1,
                               const Derived &cluster2) {
    Admissibility retval;
    const value_type dist = computeDistance(cluster1, cluster2);
    const value_type row_radius = 0.5 * cluster1.bb().col(2).norm();
    const value_type col_radius = 0.5 * cluster2.bb().col(2).norm();
    const value_type radius = row_radius > col_radius ? row_radius : col_radius;

    if (radius > eta_ * dist) {
      // check if either cluster is a leaf in that case,
      // compute the full matrix block
      if (!cluster1.nSons() || !cluster2.nSons())
        return Dense;
      else
        return Refine;
    } else
      return LowRank;
  }
  //////////////////////////////////////////////////////////////////////////////
  const std::vector<Eigen::Triplet<value_type>> &pattern_triplets() const {
    return triplet_list_;
  }
  eigenMatrix matrix() const {
    Eigen::SparseMatrix<value_type> S(N_, N_);
    S.setFromTriplets(triplet_list_.begin(), triplet_list_.end());
    return eigenMatrix(S);
  }

private:
  /**
   *  \brief recursively computes for a given pair of row and column clusters
   *         the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi, A^SigmaSigma]
   **/
  template <typename EntryGenerator>
  eigenMatrix recursivelyComputeBlock(const Derived &TR, const Derived &TC,
                                      const EntryGenerator &e_gen) {
    eigenMatrix buf(0, 0);
    eigenMatrix retval(0, 0);
    ++compute_block_calls_;
    // check for admissibility
    if (compareCluster(TR, TC) == LowRank) {
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
          eigenMatrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
          buf.conservativeResize(ret.rows(), buf.cols() + TC.sons(j).nscalfs());
          buf.rightCols(TC.sons(j).nscalfs()) =
              ret.leftCols(TC.sons(j).nscalfs());
        }
        retval = buf * TC.Q();
      } else if (TR.nSons() && !TC.nSons()) {
        // the col cluster is a leaf cluster: recursion on the row cluster
        for (auto i = 0; i < TR.nSons(); ++i) {
          eigenMatrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
          buf.conservativeResize(ret.cols(), buf.cols() + TR.sons(i).nscalfs());
          buf.rightCols(TR.sons(i).nscalfs()) =
              ret.transpose().leftCols(TR.sons(i).nscalfs());
        }
        retval = (buf * TR.Q()).transpose();
      } else {
        // neither is a leaf, let recursion handle this
        for (auto i = 0; i < TR.nSons(); ++i) {
          eigenMatrix ret1(0, 0);
          for (auto j = 0; j < TC.nSons(); ++j) {
            eigenMatrix ret2 =
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
    eigenMatrix block(0, 0);
    eigenMatrix buf(0, 0);
    ////////////////////////////////////////////////////////////////////////////
    // if there are children of the row cluster, we proceed recursively
    if (TR.nSons() && TR.block_id() < TC.block_id()) {
      for (auto i = 0; i < TR.nSons(); ++i) {
        if (compareCluster(TR.sons(i), TC) != LowRank) {
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
          eigenMatrix ret = recursivelyComputeBlock(TR.sons(i), TC, e_gen);
          buf.conservativeResize(ret.cols(), buf.cols() + TR.sons(i).nscalfs());
          buf.rightCols(TR.sons(i).nscalfs()) =
              ret.transpose().leftCols(TR.sons(i).nscalfs());
        }
      }
      block = (buf * TR.Q()).transpose();
      // we are at a leaf of the row cluster tree
    } else {
      // if TC is a leaf, we compute the corresponding matrix block
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
            eigenMatrix ret = recursivelyComputeBlock(TR, TC.sons(j), e_gen);
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
    IndexType buff_size = 0;
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
    eigenMatrix retval;
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
  void storeBlock(IndexType srow, IndexType scol, IndexType nrows,
                  IndexType ncols,
                  const Eigen::MatrixBase<otherDerived> &block) {
    storage_size_ += ncols * nrows;
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (abs(block(j, k)) > threshold_ && srow + j <= scol + k)
          triplet_list_.push_back(
              Eigen::Triplet<value_type>(srow + j, scol + k, block(j, k)));
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> triplet_list_;
  std::vector<std::map<IndexType, eigenMatrix>> buffer_;
  ProgressBar PB_;
  value_type eta_;
  value_type threshold_;
  IndexType N_;
  size_t storage_size_;
  size_t max_buff_size_;
  IndexType compute_block_calls_;
};
} // namespace FMCA
#endif

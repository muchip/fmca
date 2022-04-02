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
#ifndef FMCA_SAMPLETS_SPARSECOMPRESSOR_H_
#define FMCA_SAMPLETS_SPARSECOMPRESSOR_H_

#include "../util/SparseMatrix.h"
namespace FMCA {

template <typename Derived> struct sparse_compressor_impl {
  enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;

  template <typename EntryGenerator>
  void compress(SampletTreeBase<Derived> &ST, const EntryGenerator &M) {
    eigen_assert(ST.is_root() && "compress needs to be called from root");
    triplet_list_.clear();
    const IndexType n_samplet_blocks = std::distance(ST.cbegin(), ST.cend());
    storage_size_ = 0;
    ////////////////////////////////////////////////////////////////////////////
    // set up the compressed matrix
    for (auto &&col : ST.derived())
      for (auto &&row : ST.derived()) {
        eigenMatrix block = recursivelyComputeBlock(row, col, M);
        if (row.nsamplets() && col.nsamplets() && !row.is_root() &&
            !col.is_root())
          storeBlock(row.start_index(), col.start_index(), row.nsamplets(),
                     col.nsamplets(),
                     block.bottomRightCorner(row.nsamplets(), col.nsamplets()));
      }
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

    if (dist == 0) {
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
    eigenMatrix retval = eigenMatrix::Zero(TR.Q().cols(), TC.Q().cols());
    // check for admissibility
    if (compareCluster(TR, TC) == LowRank) {
      // retval = eigenMatrix::Zero(TR.V().cols(), TC.V().cols());
    } else {
      if (!TR.nSons() && !TC.nSons()) {
        // both are leafs: compute the block and return
        buf.resize(TR.indices().size(), TC.indices().size());
        buf.setZero();
        for (auto j = 0; j < TC.indices().size(); ++j)
          for (auto i = 0; i < TR.indices().size(); ++i)
            buf(i, j) = e_gen(TR.indices()[i], TC.indices()[j]);
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
  template <typename otherDerived>
  void storeBlock(IndexType srow, IndexType scol, IndexType nrows,
                  IndexType ncols,
                  const Eigen::MatrixBase<otherDerived> &block) {
    storage_size_ += ncols * nrows;
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (abs(block(j, k)) > 1e-15)
          triplet_list_.push_back(
              Eigen::Triplet<value_type>(srow + j, scol + k, block(j, k)));
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> triplet_list_;
  IndexType N_;
  size_t storage_size_;
};
} // namespace FMCA
#endif

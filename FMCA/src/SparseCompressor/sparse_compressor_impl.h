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

  template <typename EntryGenerator>
  void compress(SampletTreeBase<Derived> &ST, const EntryGenerator &M) {
    eigen_assert(ST.is_root() && "compress needs to be called from root");
    triplet_list_.clear();
    const Index n_samplet_blocks = std::distance(ST.cbegin(), ST.cend());
    storage_size_ = 0;
    ////////////////////////////////////////////////////////////////////////////
    // set up the compressed matrix
    for (auto &&col : ST.derived())
      for (auto &&row : ST.derived()) {
        Matrix block = recursivelyComputeBlock(row, col, M);
        if (row.nsamplets() && col.nsamplets() && !row.is_root() &&
            !col.is_root())
          storeBlock(row.start_index(), col.start_index(), row.nsamplets(),
                     col.nsamplets(),
                     block.bottomRightCorner(row.nsamplets(), col.nsamplets()));
      }
  }
  //////////////////////////////////////////////////////////////////////////////
  Scalar computeDistance(const Derived &TR, const Derived &TC) {
    const Scalar row_radius = 0.5 * TR.bb().col(2).norm();
    const Scalar col_radius = 0.5 * TC.bb().col(2).norm();
    const Scalar dist = 0.5 * (TR.bb().col(0) - TC.bb().col(0) +
                                   TR.bb().col(1) - TC.bb().col(1))
                                      .norm() -
                            row_radius - col_radius;
    return dist > 0 ? dist : 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  Admissibility compareCluster(const Derived &cluster1,
                               const Derived &cluster2) {
    Admissibility retval;
    const Scalar dist = computeDistance(cluster1, cluster2);

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
  const std::vector<Eigen::Triplet<Scalar>> &pattern_triplets() const {
    return triplet_list_;
  }
  Matrix matrix() const {
    Eigen::SparseMatrix<Scalar> S(N_, N_);
    S.setFromTriplets(triplet_list_.begin(), triplet_list_.end());
    return Matrix(S);
  }

private:
  /**
   *  \brief recursively computes for a given pair of row and column clusters
   *         the four blocks [A^PhiPhi, A^PhiSigma; A^SigmaPhi, A^SigmaSigma]
   **/
     template <typename EntryGenerator>
   Matrix recursivelyComputeBlock(const Derived &TR, const Derived &TC,
                                  const EntryGenerator &e_gen) {
     Matrix buf(0, 0);
     // check for admissibility
     if (compareCluster(TR, TC) == LowRank) {
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
  //////////////////////////////////////////////////////////////////////////////
  template <typename otherDerived>
  void storeBlock(Index srow, Index scol, Index nrows,
                  Index ncols,
                  const Eigen::MatrixBase<otherDerived> &block) {
    storage_size_ += ncols * nrows;
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < nrows; ++j)
        if (abs(block(j, k)) > 1e-15)
          triplet_list_.push_back(
              Eigen::Triplet<Scalar>(srow + j, scol + k, block(j, k)));
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<Scalar>> triplet_list_;
  Index N_;
  size_t storage_size_;
};
} // namespace FMCA
#endif

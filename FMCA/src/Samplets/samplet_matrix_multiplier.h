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
#ifndef FMCA_SAMPLETS_SAMPLETMATRIXMULTIPLIER_H_
#define FMCA_SAMPLETS_SAMPLETMATRIXMULTIPLIER_H_
namespace FMCA {

template <typename Derived>
struct samplet_matrix_multiplier {
  enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;

  Eigen::SparseMatrix<value_type> multiply(
      SampletTreeBase<Derived> &ST, const Eigen::SparseMatrix<value_type> &M1,
      const Eigen::SparseMatrix<value_type> &M2, value_type eta = 0.8,
      value_type threshold = 1e-6) {
    Eigen::SparseMatrix<value_type> retval(M1.rows(), M1.cols());
    eigen_assert(ST.is_root() && "compress needs to be called from root");
    N_ = ST.indices().size();
    threshold_ = threshold;
    eta_ = eta;
    setupColumn(ST.derived(), ST.derived(), M1, M2);
    // set up remainder of the first column
    for (const auto &it : ST) {
      if (!it.is_root()) {
        storeBlock(it.derived().start_index(), ST.derived().start_index(),
                   it.derived().nsamplets(), ST.derived().Q().cols(), M1, M2);
      }
    }
    retval.setFromTriplets(triplet_list_.begin(), triplet_list_.end());
    triplet_list_.clear();
    return retval;
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
  //////////////////////////////////////////////////////////////////////////////
  void setupRow(const Derived &TR, const Derived &TC,
                const Eigen::SparseMatrix<value_type> &M1,
                const Eigen::SparseMatrix<value_type> &M2) {
    ////////////////////////////////////////////////////////////////////////////
    // if there are children of the row cluster, call their multiply as well
    if (TR.nSons())
      for (auto i = 0; i < TR.nSons(); ++i)
        if (compareCluster(TR.sons(i), TC) != LowRank)
          setupRow(TR.sons(i), TC, M1, M2);
    if (TR.nsamplets() && TC.nsamplets() && !TC.is_root() && !TR.is_root())
      storeBlock(TR.start_index(), TC.start_index(), TR.nsamplets(),
                 TC.nsamplets(), M1, M2);
  }
  //////////////////////////////////////////////////////////////////////////////
  void setupColumn(const Derived &TR, const Derived &TC,
                   const Eigen::SparseMatrix<value_type> &M1,
                   const Eigen::SparseMatrix<value_type> &M2) {
    if (TC.nSons())
      for (auto i = 0; i < TC.nSons(); ++i) setupColumn(TR, TC.sons(i), M1, M2);
    setupRow(TR, TC, M1, M2);
    if (TC.is_root())
      storeBlock(TR.start_index(), TC.start_index(), TR.Q().cols(),
                 TC.Q().cols(), M1, M2);
    // otherwise store  [A^PhiSigma; A^SigmaSigma]
    else if (TC.nsamplets())
      storeBlock(TR.start_index(), TC.start_index(), TR.Q().cols(),
                 TC.nsamplets(), M1, M2);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void storeBlock(IndexType srow, IndexType scol, IndexType nrows,
                  IndexType ncols, const Eigen::SparseMatrix<value_type> &M1,
                  const Eigen::SparseMatrix<value_type> &M2) {
    Eigen::SparseMatrix<value_type> block =
        M1.middleCols(srow, nrows).transpose() * M2.middleCols(scol, ncols);
    for (auto i = 0; i < block.outerSize(); ++i)
      for (typename Eigen::SparseMatrix<value_type>::InnerIterator it(block, i);
           it; ++it) {
        if (abs(it.value()) > threshold_)
          triplet_list_.push_back(Eigen::Triplet<value_type>(
              srow + it.row(), scol + it.col(), it.value()));
      }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member variables
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<value_type>> triplet_list_;
  value_type eta_;
  value_type threshold_;
  IndexType N_;
};
}  // namespace FMCA
#endif

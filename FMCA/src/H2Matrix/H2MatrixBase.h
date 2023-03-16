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
#ifndef FMCA_H2MATRIX_H2MATRIXBASE_H_
#define FMCA_H2MATRIX_H2MATRIXBASE_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief H2MatrixNodeBase defines the basic fields required for an
 *         abstract H2Matrix
 **/
template <typename Derived>
struct H2MatrixNodeBase : public NodeBase<Derived> {
  H2MatrixNodeBase()
      : row_cluster_(nullptr), col_cluster_(nullptr), is_low_rank_(false) {
    S_.resize(0, 0);
  }
  Matrix S_;
  const typename internal::traits<Derived>::RowCType *row_cluster_;
  const typename internal::traits<Derived>::ColCType *col_cluster_;
  Index nrclusters_;
  Index ncclusters_;
  bool is_low_rank_;
};

/**
 *  \ingroup H2Matrix
 *  \brief The H2Matrix class manages H2 matrices for a given
 *         H2ClusterTree.

 */
template <typename Derived>
struct H2MatrixBase : TreeBase<Derived> {
  typedef TreeBase<Derived> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::dad;
  using Base::derived;
  using Base::init;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  typedef typename internal::traits<Derived>::RowCType RowCType;
  typedef typename internal::traits<Derived>::ColCType ColCType;
  //////////////////////////////////////////////////////////////////////////////
  // base class methods
  //////////////////////////////////////////////////////////////////////////////
  Index rows() const { return (node().row_cluster_)->indices().size(); }
  Index cols() const { return (node().col_cluster_)->indices().size(); }
  Index nrclusters() const { return node().nrclusters_; }
  Index ncclusters() const { return node().ncclusters_; }
  const RowCType *rcluster() const { return node().row_cluster_; }
  const ColCType *ccluster() const { return node().col_cluster_; }
  bool is_low_rank() const { return node().is_low_rank_; }
  const Matrix &matrixS() const { return node().S_; }
  //////////////////////////////////////////////////////////////////////////////
  // (m, n, fblocks, lrblocks, nz(A), mem)
  Matrix statistics() const {
    Matrix retval(6, 1);
    Index low_rank_blocks = 0;
    Index full_blocks = 0;
    Index memory = 0;
    assert(is_root() && "statistics needs to be called from root");
    for (auto &&it : *this) {
      if (!it.nSons()) {
        if (it.is_low_rank())
          ++low_rank_blocks;
        else
          ++full_blocks;
        memory += it.node().S_.size();
      }
    }
    std::cout << "matrix size:                  " << rows() << " x " << cols()
              << std::endl;
    retval(0, 0) = rows();
    retval(1, 0) = cols();
    std::cout << "number of low rank blocks:    " << low_rank_blocks
              << std::endl;
    retval(2, 0) = low_rank_blocks;
    std::cout << "number of full blocks:        " << full_blocks << std::endl;
    retval(3, 0) = full_blocks;
    std::cout << "nz per row:                   "
              << round(Scalar(memory) / (node().col_cluster_)->indices().size())
              << std::endl;
    retval(4, 0) =
        round(Scalar(memory) / (node().col_cluster_)->indices().size());
    std::cout << "storage size:                 "
              << Scalar(memory * sizeof(Scalar)) / 1e9 << "GB" << std::endl;
    retval(5, 0) = Scalar(memory * sizeof(Scalar)) / 1e9;
    return retval;
  }

  Matrix operator*(const Matrix &rhs) const {
    return internal::matrix_vector_product_impl(*this, rhs);
  }
  //////////////////////////////////////////////////////////////////////////////
  Matrix full() const {
    assert(is_root() && "full needs to be called from root");
    Matrix I(cols(), cols());
    return *this * I;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void computeH2Matrix(const RowCType &CT1, const ColCType &CT2,
                       const MatrixEvaluator &mat_eval, Scalar eta) {
    node().row_cluster_ = &CT1;
    node().col_cluster_ = &CT2;
    const Admissibility adm = compareCluster(CT1, CT2, eta);
    if (adm == LowRank) {
      node().is_low_rank_ = true;
      mat_eval.interpolate_kernel(CT1, CT2, std::addressof(node().S_));
    } else if (adm == Refine) {
      appendSons(CT1.nSons() * CT2.nSons());
      for (auto j = 0; j < CT2.nSons(); ++j)
        for (auto i = 0; i < CT1.nSons(); ++i) {
          sons(i + j * CT1.nSons())
              .computeH2Matrix(CT1.sons(i), CT2.sons(j), mat_eval, eta);
        }
    } else {
      mat_eval.compute_dense_block(CT1, CT2, std::addressof(node().S_));
    }
    return;
  }
};

}  // namespace FMCA
#endif

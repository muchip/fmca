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
  typedef typename internal::traits<Derived>::Node Node;
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
    Index lr_blocks = 0;
    Index f_blocks = 0;
    Index mem = 0;
    assert(is_root() && "statistics needs to be called from root");
    for (auto &&it : *this) {
      if (!it.nSons()) {
        if (it.is_low_rank())
          ++lr_blocks;
        else
          ++f_blocks;
        mem += it.node().S_.size();
      }
    }
    retval(0, 0) = rows();
    retval(1, 0) = cols();
    retval(2, 0) = lr_blocks;
    retval(3, 0) = f_blocks;
    retval(4, 0) = round(Scalar(mem) / (node().col_cluster_)->indices().size());
    retval(5, 0) = Scalar(mem * sizeof(Scalar)) / 1e9;
    std::cout << "matrix size:                  " << rows() << " x " << cols()
              << std::endl;
    std::cout << "number of low rank blocks:    " << lr_blocks << std::endl;
    std::cout << "number of full blocks:        " << f_blocks << std::endl;
    std::cout << "nz per row:                   " << retval(4, 0) << std::endl;
    std::cout << "storage size:                 " << retval(5, 0) << "GB"
              << std::endl;
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
  void computePattern(const RowCType &CT1, const ColCType &CT2, Scalar eta) {
    if (CT1.is_root() && CT2.is_root()) {
      node().nrclusters_ = std::distance(CT1.cbegin(), CT1.cend());
      node().ncclusters_ = std::distance(CT2.cbegin(), CT2.cend());
    }
    node().row_cluster_ = &CT1;
    node().col_cluster_ = &CT2;
    const Admissibility adm = compareCluster(CT1, CT2, eta);
    if (adm == LowRank) {
      node().is_low_rank_ = true;
    } else if (adm == Refine) {
      appendSons(CT1.nSons() * CT2.nSons());
      for (Index j = 0; j < CT2.nSons(); ++j)
        for (Index i = 0; i < CT1.nSons(); ++i) {
          sons(i + j * CT1.nSons())
              .computePattern(CT1.sons(i), CT2.sons(j), eta);
        }
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void computeH2Matrix(const RowCType &CT1, const ColCType &CT2,
                       const MatrixEvaluator &mat_eval, Scalar eta) {
    computePattern(CT1, CT2, eta);
    std::vector<Node *> queue;
    for (auto &&it : *this)
      if (!it.nSons()) queue.push_back(std::addressof(it.node()));
#pragma omp parallel for
    for (Index i = 0; i < queue.size(); ++i) {
      const RowCType &row = *(queue[i]->row_cluster_);
      const ColCType &col = *(queue[i]->col_cluster_);
      if (queue[i]->is_low_rank_)
        mat_eval.interpolate_kernel(row, col, std::addressof(queue[i]->S_));
      else
        mat_eval.compute_dense_block(row, col, std::addressof(queue[i]->S_));
    }
    return;
  }
#if 0
  template <typename MatrixEvaluator>
  void computeH2Matrix(const RowCType &CT1, const ColCType &CT2,
                       const MatrixEvaluator &mat_eval, Scalar eta) {
    computePattern(CT1, CT2, eta);
    for (auto &&it : *this)
      if (!it.nSons()) {
        const RowCType &row = *(it.node().row_cluster_);
        const ColCType &col = *(it.node().col_cluster_);
        if (it.is_low_rank())
          mat_eval.interpolate_kernel(row, col, std::addressof(it.node().S_));
        else
          mat_eval.compute_dense_block(row, col, std::addressof(it.node().S_));
      }
    return;
  }
#endif
};

}  // namespace FMCA
#endif

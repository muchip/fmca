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
  Matrix &matrixS() { return node().S_; }
  //////////////////////////////////////////////////////////////////////////////
  // (m, n, fblocks, lrblocks, nz(A), mem)
  Matrix statistics() const {
    Matrix retval(6, 1);
    size_t lr_blocks = 0;
    size_t f_blocks = 0;
    size_t mem = 0;
    assert(is_root() && "statistics needs to be called from root");
    for (auto &&it : *this) {
      const RowCType &row = *(it.rcluster());
      const ColCType &col = *(it.ccluster());
      if (!it.nSons()) {
        if (it.is_low_rank()) {
          const Index brows = row.nSons() ? row.Es()[0].rows() : row.V().rows();
          const Index bcols = col.nSons() ? col.Es()[0].rows() : col.V().rows();
          ++lr_blocks;
          mem += brows * bcols;
        } else {
          ++f_blocks;
          mem += row.indices().size() * row.indices().size();
        }
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

  template <typename MatrixEvaluator>
  Matrix action(const MatrixEvaluator &mat_eval, const Matrix &rhs) const {
    std::vector<std::vector<const Derived *>> scheduler;
    Matrix lhs(rows(), rhs.cols());
    // forward transform righ hand side
    std::vector<Matrix> trhs = internal::forward_transform_impl(*this, rhs);
    std::cout << "forward transform " << std::endl;
    std::vector<Matrix> tlhs(nrclusters());
    for (const auto &it : *(rcluster())) {
      if (it.nSons())
        tlhs[it.block_id()].resize(it.Es()[0].rows(), rhs.cols());
      else
        tlhs[it.block_id()].resize(it.V().rows(), rhs.cols());
      tlhs[it.block_id()].setZero();
    }
    std::cout << "tlhs" << std::endl;
    scheduler.resize(ncclusters());
    for (const auto &it : *this)
      if (!it.nSons())
        scheduler[it.ccluster()->block_id()].push_back(std::addressof(it));
    std::cout << "scheduler" << std::endl;
    for (const auto &it2 : scheduler) {
#pragma omp parallel for schedule(dynamic)
      for (Index k = 0; k < it2.size(); ++k) {
        Matrix S;
        const Derived &it = *(it2[k]);
        const Index i = it.rcluster()->block_id();
        const Index j = it.ccluster()->block_id();
        const Index ii = (it.rcluster())->indices_begin();
        const Index jj = (it.ccluster())->indices_begin();
        if (it.is_low_rank()) {
          mat_eval.interpolate_kernel(*(it.rcluster()), *(it.ccluster()), &S);
          tlhs[i] += S * trhs[j];
        } else {
          mat_eval.compute_dense_block(*(it.rcluster()), *(it.ccluster()), &S);
          lhs.middleRows(ii, S.rows()) += S * rhs.middleRows(jj, S.cols());
        }
      }
    }
    // backward transform left hand side
    internal::backward_transform_recursion(*(rcluster()), &lhs, tlhs);
    return lhs;
  }

  //////////////////////////////////////////////////////////////////////////////
  Matrix full() const {
    assert(is_root() && "full needs to be called from root");
    Matrix I(cols(), cols());
    return *this * I;
  }
  void computePattern(const RowCType &CT1, const ColCType &CT2, Scalar eta) {
    node().nrclusters_ = std::distance(CT1.cbegin(), CT1.cend());
    node().ncclusters_ = std::distance(CT2.cbegin(), CT2.cend());
    node().row_cluster_ = std::addressof(CT1);
    node().col_cluster_ = std::addressof(CT2);
    std::vector<Derived *> stack;
    stack.push_back(std::addressof(this->derived()));
    while (stack.size()) {
      Derived *block = stack.back();
      stack.pop_back();
      const RowCType &row = *(block->rcluster());
      const ColCType &col = *(block->ccluster());
      const Admissibility adm = compareCluster(row, col, eta);
      if (adm == LowRank) {
        const Index brows = row.nSons() ? row.Es()[0].rows() : row.V().rows();
        const Index bcols = col.nSons() ? col.Es()[0].rows() : col.V().rows();
        // only use low-rank if it is actually cheaper than storing the block
        (block->node()).is_low_rank_ =
            (brows * bcols < row.indices().size() * col.indices().size());
      } else if (adm == Refine) {
        block->appendSons(row.nSons() * col.nSons());
        for (Index j = 0; j < col.nSons(); ++j)
          for (Index i = 0; i < row.nSons(); ++i) {
            Derived &son = block->sons(i + j * row.nSons());
            son.node().row_cluster_ = std::addressof(row.sons(i));
            son.node().col_cluster_ = std::addressof(col.sons(j));
            stack.push_back(std::addressof(son));
          }
      }
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void computeH2Matrix(const RowCType &CT1, const ColCType &CT2,
                       const MatrixEvaluator &mat_eval, Scalar eta) {
    computePattern(CT1, CT2, eta);
    Index pos = 0;
#pragma omp parallel shared(pos)
    {
      Index i = 0;
      Index prev_i = 0;
      typename Base::iterator it = this->begin();
#pragma omp atomic capture
      i = pos++;
      while (it != this->end()) {
        Index dist = i - prev_i;
        while (dist > 0 && it != this->end()) {
          --dist;
          ++it;
        }
        if (it == this->end()) break;
        if (!(it->nSons())) {
          const RowCType &row = *(it->rcluster());
          const ColCType &col = *(it->ccluster());
          if (it->is_low_rank())
            mat_eval.interpolate_kernel(row, col, &(it->matrixS()));
          else
            mat_eval.compute_dense_block(row, col, &(it->matrixS()));
        }
        prev_i = i;
#pragma omp atomic capture
        i = pos++;
      }
    }
    return;
  }
};

}  // namespace FMCA
#endif

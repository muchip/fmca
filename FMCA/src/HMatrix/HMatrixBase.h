// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2024, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_HMATRIX_HMATRIXBASE_H_
#define FMCA_HMATRIX_HMATRIXBASE_H_

namespace FMCA {

/**
 *  \ingroup HMatrix
 *  \brief HMatrixNodeBase defines the basic fields required for an
 *         abstract HMatrix
 **/
template <typename Derived>
struct HMatrixNodeBase : public NodeBase<Derived> {
  HMatrixNodeBase()
      : row_cluster_(nullptr), col_cluster_(nullptr), is_low_rank_(false) {
    F_.resize(0, 0);
    L_.resize(0, 0);
    R_.resize(0, 0);
  }
  Matrix F_;
  Matrix L_;
  Matrix R_;

  const typename internal::traits<Derived>::RowCType *row_cluster_;
  const typename internal::traits<Derived>::ColCType *col_cluster_;
  Index nrclusters_;
  Index ncclusters_;
  bool is_low_rank_;
};

/**
 *  \ingroup HMatrix
 *  \brief The HMatrixBase class is the common base class for all H2 matrices
 */
template <typename Derived>
struct HMatrixBase : TreeBase<Derived> {
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
  Index rows() const { return (node().row_cluster_)->block_size(); }
  Index cols() const { return (node().col_cluster_)->block_size(); }
  Index nrclusters() const { return node().nrclusters_; }
  Index ncclusters() const { return node().ncclusters_; }
  const RowCType *rcluster() const { return node().row_cluster_; }
  const ColCType *ccluster() const { return node().col_cluster_; }
  bool is_low_rank() const { return node().is_low_rank_; }
  const Matrix &matrixF() const { return node().F_; }
  Matrix &matrixF() { return node().F_; }
  const Matrix &matrixL() const { return node().L_; }
  Matrix &matrixL() { return node().L_; }
  const Matrix &matrixR() const { return node().R_; }
  Matrix &matrixR() { return node().R_; }

  //////////////////////////////////////////////////////////////////////////////
  // (m, n, fblocks, lrblocks, nz(A), mem)
  Matrix statistics() const {
    Matrix retval(6, 1);
    size_t lr_blocks = 0;
    size_t f_blocks = 0;
    size_t mem = 0;
    assert(is_root() && "statistics needs to be called from root");
    for (const auto &it : *this) {
      if (!it.nSons()) {
        if (it.is_low_rank()) {
          ++lr_blocks;
          mem += it.matrixL().size() + it.matrixR().size();
        } else {
          ++f_blocks;
          mem += it.matrixF().size();
        }
      }
    }
    retval(0, 0) = rows();
    retval(1, 0) = cols();
    retval(2, 0) = lr_blocks;
    retval(3, 0) = f_blocks;
    retval(4, 0) = round(Scalar(mem) / (node().row_cluster_)->block_size());
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
    Matrix retval(rows(), rhs.cols());
    retval.setZero();
    Index pos = 0;
#pragma omp parallel shared(pos)
    {
      Matrix loc_ret(rows(), rhs.cols());
      loc_ret.setZero();
      Index i = 0;
      Index prev_i = 0;
      auto it = this->begin();
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
          if (it->is_low_rank()) {
            loc_ret.middleRows(row.indices_begin(), row.block_size()) +=
                it->matrixL() *
                (it->matrixR().transpose() *
                 rhs.middleRows(col.indices_begin(), col.block_size()))
                    .eval();

          } else {
            loc_ret.middleRows(row.indices_begin(), row.block_size()) +=
                it->matrixF() *
                rhs.middleRows(col.indices_begin(), col.block_size());
          }
        }
        prev_i = i;
#pragma omp atomic capture
        i = pos++;
      }
#pragma omp critical
      retval += loc_ret;
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  Matrix full() const {
    assert(is_root() && "full needs to be called from root");
    return *this * Matrix::Identity(cols(), cols());
  }
  //////////////////////////////////////////////////////////////////////////////
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
      const Admissibility adm = Derived::CC::compare(row, col, eta);
      if (adm == LowRank) {
        (block->node()).is_low_rank_ = true;
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
  void computeHMatrix(const RowCType &CT1, const ColCType &CT2,
                      const MatrixEvaluator &mat_eval, const Scalar eta,
                      const Scalar prec = 1e-6) {
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
          if (it->is_low_rank()) {
            ACA(mat_eval, row, col, &(it->matrixL()), &(it->matrixR()), prec);
          } else {
            mat_eval.compute_dense_block(row, col, &(it->matrixF()));
          }
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

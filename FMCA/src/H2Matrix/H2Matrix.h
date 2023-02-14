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
#ifndef FMCA_H2MATRIX_H2MATRIX_H_
#define FMCA_H2MATRIX_H2MATRIX_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief The H2Matrix class manages H2 matrices for a given
 *         H2ClusterTree.

 */
template <typename Derived>
class H2Matrix {
 public:
  using iterator = IDDFSForwardIterator<H2Matrix, false>;
  using const_iterator = IDDFSForwardIterator<H2Matrix, true>;
  friend iterator;
  friend const_iterator;

  iterator begin() { return iterator(static_cast<H2Matrix *>(this), 0); }
  iterator end() { return iterator(nullptr, 0); }

  const_iterator begin() const {
    return const_iterator(static_cast<const H2Matrix *>(this), 0);
  }
  const_iterator end() const { return const_iterator(nullptr, 0); }

  const_iterator cbegin() const {
    return const_iterator(static_cast<const H2Matrix *>(this), 0);
  }
  const_iterator cend() const { return const_iterator(nullptr, 0); }
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2Matrix() : is_low_rank_(false), dad_(nullptr), level_(0) {}
  template <typename EntryGenerator>
  H2Matrix(const H2ClusterTreeBase<Derived> &CT, const EntryGenerator &e_gen,
           Scalar eta = 0.8)
      : is_low_rank_(false), dad_(nullptr), level_(0) {
    init(CT, e_gen, eta);
  }

  template <typename EntryGenerator>
  H2Matrix(const H2ClusterTreeBase<Derived> &RCT,
           const H2ClusterTreeBase<Derived> &CCT, const EntryGenerator &e_gen,
           Scalar eta = 0.8)
      : is_low_rank_(false), dad_(nullptr), level_(0) {
    init(RCT, CCT, e_gen, eta);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void init(const H2ClusterTreeBase<Derived> &CT,
            const MatrixEvaluator &mat_eval, Scalar eta = 0.8) {
    ncclusters_ = std::distance(CT.cbegin(), CT.cend());
    nrclusters_ = ncclusters_;
    computeH2Matrix(CT.derived(), CT.derived(), mat_eval, eta);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void init(const H2ClusterTreeBase<Derived> &RCT,
            const H2ClusterTreeBase<Derived> &CCT,
            const MatrixEvaluator &mat_eval, Scalar eta = 0.8) {
    ncclusters_ = std::distance(CCT.cbegin(), CCT.cend());
    nrclusters_ = std::distance(RCT.cbegin(), CCT.cend());
    computeH2Matrix(RCT.derived(), CCT.derived(), mat_eval, eta);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // init block cluster tree
  //////////////////////////////////////////////////////////////////////////////
  void initBlockClusterTree(const Derived &CT1, const Derived &CT2,
                            Scalar eta) {
    if (CT1.is_root() && CT2.is_root()) {
      ncclusters_ = std::distance(CT2.cbegin(), CT2.cend());
      nrclusters_ = std::distance(CT1.cbegin(), CT1.cend());
    }
    row_cluster_ = &CT1;
    col_cluster_ = &CT2;
    level_ = CT1.level();
    Admissibility adm = compareCluster(CT1, CT2, eta);
    if (adm == LowRank) {
      is_low_rank_ = true;
    } else if (adm == Refine) {
      sons_.resize(CT1.nSons(), CT2.nSons());
      for (auto j = 0; j < CT2.nSons(); ++j)
        for (auto i = 0; i < CT1.nSons(); ++i) {
          sons_(i, j).dad_ = this;
          sons_(i, j).initBlockClusterTree(CT1.sons(i), CT2.sons(j), eta);
        }
    } else {
      is_low_rank_ = false;
    }
    return;
  }

  Index rows() const { return row_cluster_->indices().size(); }
  Index cols() const { return col_cluster_->indices().size(); }
  Index level() const { return level_; }
  Index ncclusters() const { return ncclusters_; }
  Index nrclusters() const { return nrclusters_; }
  const Derived *rcluster() const { return row_cluster_; }
  const Derived *ccluster() const { return col_cluster_; }
  const GenericMatrix<H2Matrix> &sons() const { return sons_; }
  bool is_low_rank() const { return is_low_rank_; }
  const Matrix &matrixS() const { return S_; }
  //////////////////////////////////////////////////////////////////////////////
  // (m, n, fblocks, lrblocks, nz(A), mem)
  Matrix get_statistics() const {
    Matrix retval(6, 1);
    Index low_rank_blocks = 0;
    Index full_blocks = 0;
    Index memory = 0;
    getStatisticsRecursion(&low_rank_blocks, &full_blocks, &memory);
    std::cout << "matrix size:                  "
              << row_cluster_->indices().size() << " x "
              << col_cluster_->indices().size() << std::endl;
    retval(0, 0) = row_cluster_->indices().size();
    retval(1, 0) = col_cluster_->indices().size();
    std::cout << "number of low rank blocks:    " << low_rank_blocks
              << std::endl;
    retval(2, 0) = low_rank_blocks;
    std::cout << "number of full blocks:        " << full_blocks << std::endl;
    retval(3, 0) = full_blocks;
    std::cout << "nz per row:                   "
              << round(Scalar(memory) / col_cluster_->indices().size())
              << std::endl;
    retval(4, 0) = round(Scalar(memory) / col_cluster_->indices().size());
    std::cout << "storage size:                 "
              << Scalar(memory * sizeof(Scalar)) / 1e9 << "GB" << std::endl;
    retval(5, 0) = Scalar(memory * sizeof(Scalar)) / 1e9;
    return retval;
  }

  template <typename otherDerived>
  Matrix operator*(const Eigen::MatrixBase<otherDerived> &rhs) const {
    return internal::matrix_vector_product_impl(*this, rhs);
  }
  //////////////////////////////////////////////////////////////////////////////
  Matrix full() const {
    eigen_assert(row_cluster_->is_root() && col_cluster_->is_root() &&
                 "method needs to be called from the root");
    Matrix retval(row_cluster_->indices().size(),
                  row_cluster_->indices().size());
    computeFullMatrixRecursion(*row_cluster_, *row_cluster_, &retval);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  static Scalar computeDistance(const Derived &cluster1,
                                const Derived &cluster2) {
    const Scalar row_radius = 0.5 * cluster1.bb().col(2).norm();
    const Scalar col_radius = 0.5 * cluster2.bb().col(2).norm();
    const Scalar dist = 0.5 * (cluster1.bb().col(0) - cluster2.bb().col(0) +
                               cluster1.bb().col(1) - cluster2.bb().col(1))
                                  .norm() -
                        row_radius - col_radius;
    return dist > 0 ? dist : 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  static Admissibility compareCluster(const Derived &cluster1,
                                      const Derived &cluster2, Scalar eta) {
    Admissibility retval;
    const Scalar dist = computeDistance(cluster1, cluster2);
    const Scalar row_radius = 0.5 * cluster1.bb().col(2).norm();
    const Scalar col_radius = 0.5 * cluster2.bb().col(2).norm();
    const Scalar radius = row_radius > col_radius ? row_radius : col_radius;

    if (radius > eta * dist) {
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
 private:
  //////////////////////////////////////////////////////////////////////////////
  void getStatisticsRecursion(Index *low_rank_blocks, Index *full_blocks,
                              Index *memory) const {
    if (sons_.size()) {
      for (const auto &s : sons_)
        s.getStatisticsRecursion(low_rank_blocks, full_blocks, memory);
    } else {
      if (is_low_rank_)
        ++(*low_rank_blocks);
      else
        ++(*full_blocks);
      (*memory) += S_.size();
    }
    return;
  }

  void computeFullMatrixRecursion(const Derived &CR, const Derived &CS,
                                  Matrix *target) const {
    if (sons_.size()) {
      for (auto i = 0; i < sons_.rows(); ++i)
        for (auto j = 0; j < sons_.cols(); ++j)
          sons_(i, j).computeFullMatrixRecursion(
              *(sons_(i, j).row_cluster_), *(sons_(i, j).col_cluster_), target);
    } else {
      if (is_low_rank_)
        target->block(
            row_cluster_->indices_begin(), col_cluster_->indices_begin(),
            row_cluster_->indices().size(), col_cluster_->indices().size()) =
            row_cluster_->V().transpose() * S_ * col_cluster_->V();
      else
        target->block(row_cluster_->indices_begin(),
                      col_cluster_->indices_begin(),
                      row_cluster_->indices().size(),
                      col_cluster_->indices().size()) = S_;
    }
    return;
  }

  template <typename MatrixEvaluator>
  void computeH2Matrix(const Derived &CT1, const Derived &CT2,
                       const MatrixEvaluator &mat_eval, Scalar eta) {
    row_cluster_ = &CT1;
    col_cluster_ = &CT2;
    level_ = CT1.level();
    Admissibility adm = compareCluster(CT1, CT2, eta);
    if (adm == LowRank) {
      is_low_rank_ = true;
      mat_eval.interpolate_kernel(CT1, CT2, &S_);
    } else if (adm == Refine) {
      sons_.resize(CT1.nSons(), CT2.nSons());
      for (auto j = 0; j < CT2.nSons(); ++j)
        for (auto i = 0; i < CT1.nSons(); ++i) {
          sons_(i, j).dad_ = this;
          sons_(i, j).computeH2Matrix(CT1.sons(i), CT2.sons(j), mat_eval, eta);
        }
    } else {
      mat_eval.compute_dense_block(CT1, CT2, &S_);
    }
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix<H2Matrix> sons_;
  H2Matrix *dad_;
  const Derived *row_cluster_;
  const Derived *col_cluster_;
  bool is_low_rank_;
  Matrix S_;
  Index level_;
  Index nrclusters_;
  Index ncclusters_;
};

}  // namespace FMCA
#endif

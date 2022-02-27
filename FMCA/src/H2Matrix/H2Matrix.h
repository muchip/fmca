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
#ifndef FMCA_H2MATRIX_H2MATRIX_H_
#define FMCA_H2MATRIX_H2MATRIX_H_

namespace FMCA {

/**
 *  \ingroup H2Matrix
 *  \brief The H2Matrix class manages H2 matrices for a given
 *         H2ClusterTree.

 */
template <typename Derived> class H2Matrix {
public:
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };
  using iterator = IDDFSForwardIterator<H2Matrix, false>;
  using const_iterator = IDDFSForwardIterator<H2Matrix, true>;
  friend iterator;
  friend const_iterator;

  struct RandomAccessor {
    std::vector<const H2Matrix *> nearfield;
    std::vector<std::vector<const H2Matrix *>> farfield;
  };

  iterator begin() { return iterator(static_cast<H2Matrix *>(this), 0); }
  iterator end() { return iterator(nullptr, 0); }
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
           value_type eta = 0.8)
      : is_low_rank_(false), dad_(nullptr), level_(0) {
    init(CT, e_gen, eta);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename MatrixEvaluator>
  void init(const H2ClusterTreeBase<Derived> &CT,
            const MatrixEvaluator &mat_eval, value_type eta = 0.8) {
    computeH2Matrix(CT.derived(), CT.derived(), mat_eval, eta);
    // after setting up the actual H2Matrix, we now also fill the random
    // access iterator, which allows levelwise traversal of the H2Matrix
    // and random access to the nearfield
    IndexType max_level = 0;
    nclusters_ = std::distance(CT.cbegin(), CT.cend());
    for (const auto &node : *this) {
      max_level = max_level < node.level() ? node.level() : max_level;
    }
    rnd_access_.farfield.resize(max_level + 1);
    for (const auto &node : *this) {
      if (!node.sons_.size()) {
        // we are at a leaf store pointer to the leaf in the respective array
        if (node.is_low_rank_)
          rnd_access_.farfield[node.level()].push_back(std::addressof(node));
        else
          rnd_access_.nearfield.push_back(std::addressof(node));
      }
    }
    return;
  }
  IndexType rows() const { return row_cluster_->indices().size(); }
  IndexType cols() const { return col_cluster_->indices().size(); }
  IndexType level() const { return level_; }
  IndexType nclusters() const { return nclusters_; }
  const Derived *rcluster() const { return row_cluster_; }
  const Derived *ccluster() const { return col_cluster_; }
  const GenericMatrix<H2Matrix> &sons() const { return sons_; }
  bool is_low_rank() const { return is_low_rank_; }
  const eigenMatrix &matrixS() const { return S_; }
  const RandomAccessor &rnd_accessor() const { return rnd_access_; }
  //////////////////////////////////////////////////////////////////////////////
  // (m, n, fblocks, lrblocks, nz(A), mem)
  std::vector<double> get_statistics() const {
    std::vector<double> retval;
    IndexType low_rank_blocks = 0;
    IndexType full_blocks = 0;
    IndexType memory = 0;
    getStatisticsRecursion(&low_rank_blocks, &full_blocks, &memory);
    std::cout << "matrix size: " << row_cluster_->indices().size() << " x "
              << col_cluster_->indices().size() << std::endl;
    retval.push_back(row_cluster_->indices().size());
    retval.push_back(col_cluster_->indices().size());
    std::cout << "number of low rank blocks: " << low_rank_blocks << std::endl;
    retval.push_back(low_rank_blocks);

    std::cout << "number of full blocks: " << full_blocks << std::endl;
    retval.push_back(full_blocks);

    std::cout << "nz per row: "
              << round(FloatType(memory) / col_cluster_->indices().size())
              << std::endl;
    retval.push_back(round(FloatType(memory) / col_cluster_->indices().size()));
    std::cout << "storage size: "
              << FloatType(memory * sizeof(value_type)) / 1e9 << "GB"
              << std::endl;
    retval.push_back(FloatType(memory * sizeof(value_type)) / 1e9);
    return retval;
  }

  template <typename otherDerived>
  eigenMatrix operator*(const Eigen::MatrixBase<otherDerived> &rhs) const {
    return matrix_vector_product_impl(*this, rhs);
  }
  //////////////////////////////////////////////////////////////////////////////
  eigenMatrix full() const {
    eigen_assert(row_cluster_->is_root() && col_cluster_->is_root() &&
                 "method needs to be called from the root");
    eigenMatrix retval(row_cluster_->indices().size(),
                       row_cluster_->indices().size());
    computeFullMatrixRecursion(*row_cluster_, *row_cluster_, &retval);
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  static value_type computeDistance(const Derived &cluster1,
                                    const Derived &cluster2) {
    const value_type row_radius = 0.5 * cluster1.bb().col(2).norm();
    const value_type col_radius = 0.5 * cluster2.bb().col(2).norm();
    const value_type dist = 0.5 * (cluster1.bb().col(0) - cluster2.bb().col(0) +
                                   cluster1.bb().col(1) - cluster2.bb().col(1))
                                      .norm() -
                            row_radius - col_radius;
    return dist > 0 ? dist : 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  static Admissibility compareCluster(const Derived &cluster1,
                                      const Derived &cluster2, value_type eta) {
    Admissibility retval;
    const value_type dist = computeDistance(cluster1, cluster2);
    const value_type row_radius = 0.5 * cluster1.bb().col(2).norm();
    const value_type col_radius = 0.5 * cluster2.bb().col(2).norm();
    const value_type radius = row_radius > col_radius ? row_radius : col_radius;

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
  void getStatisticsRecursion(IndexType *low_rank_blocks,
                              IndexType *full_blocks, IndexType *memory) const {
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
                                  eigenMatrix *target) const {
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
                       const MatrixEvaluator &mat_eval, value_type eta) {
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
  RandomAccessor rnd_access_;
  GenericMatrix<H2Matrix> sons_;
  H2Matrix *dad_;
  const Derived *row_cluster_;
  const Derived *col_cluster_;
  bool is_low_rank_;
  eigenMatrix S_;
  IndexType level_;
  IndexType nclusters_;
};

} // namespace FMCA
#endif

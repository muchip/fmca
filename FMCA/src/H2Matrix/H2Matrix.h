// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
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
template <typename Derived>
class H2Matrix {
 public:
  typedef typename internal::traits<Derived>::value_type value_type;
  typedef typename internal::traits<Derived>::eigenMatrix eigenMatrix;
  enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2Matrix() : is_low_rank_(false) {}
  template <typename Functor>
  H2Matrix(const eigenMatrix &P, const TreeBase<Derived> &CT,
           const Functor &fun, value_type eta = 0.8)
      : is_low_rank_(false) {
    init(P, CT, fun, eta);
  }
  //////////////////////////////////////////////////////////////////////////////
  // init
  //////////////////////////////////////////////////////////////////////////////
  template <typename Functor>
  void init(const eigenMatrix &P, const TreeBase<Derived> &CT,
            const Functor &fun, value_type eta = 0.8) {
    computeH2Matrix(P, CT.derived(), CT.derived(), fun, eta);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  void get_statistics() const {
    IndexType low_rank_blocks = 0;
    IndexType full_blocks = 0;
    IndexType memory = 0;
    getStatisticsRecursion(&low_rank_blocks, &full_blocks, &memory);
    std::cout << "matrix size: " << row_cluster_->indices().size() << " x "
              << col_cluster_->indices().size() << std::endl;
    std::cout << "number of low rank blocks: " << low_rank_blocks << std::endl;
    std::cout << "number of full blocks: " << full_blocks << std::endl;
    std::cout << "nz per row: "
              << round(FloatType(memory) / col_cluster_->indices().size())
              << std::endl;
    std::cout << "storage size: "
              << FloatType(memory * sizeof(value_type)) / 1e9 << "GB"
              << std::endl;
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
      for (auto i = 0; i < sons_.rows(); ++i)
        for (auto j = 0; j < sons_.cols(); ++j)
          sons_(i, j).getStatisticsRecursion(low_rank_blocks, full_blocks,
                                             memory);
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

  template <typename Functor>
  void computeH2Matrix(const eigenMatrix &P, const Derived &CT1,
                       const Derived &CT2, const Functor &fun, value_type eta) {
    row_cluster_ = &CT1;
    col_cluster_ = &CT2;
    Admissibility adm = compareCluster(CT1, CT2, eta);
    if (adm == LowRank) {
      const eigenMatrix &Xi = CT1.Xi();
      S_.resize(Xi.cols(), Xi.cols());
      is_low_rank_ = true;
      for (auto j = 0; j < S_.cols(); ++j)
        for (auto i = 0; i < S_.rows(); ++i)
          S_(i, j) = fun((CT1.bb().col(2).array() * Xi.col(i).array() +
                          CT1.bb().col(0).array())
                             .matrix(),
                         (CT2.bb().col(2).array() * Xi.col(j).array() +
                          CT2.bb().col(0).array())
                             .matrix());
    } else if (adm == Refine) {
      sons_.resize(CT1.nSons(), CT2.nSons());
      for (auto j = 0; j < CT2.nSons(); ++j)
        for (auto i = 0; i < CT1.nSons(); ++i)
          sons_(i, j).computeH2Matrix(P, CT1.sons(i), CT2.sons(j), fun, eta);
    } else {
      S_.resize(CT1.indices().size(), CT2.indices().size());
      for (auto j = 0; j < CT2.indices().size(); ++j)
        for (auto i = 0; i < CT1.indices().size(); ++i)
          S_(i, j) = fun(P.col(CT1.indices()[i]), P.col(CT2.indices()[j]));
    }
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix<H2Matrix> sons_;
  const Derived *row_cluster_;
  const Derived *col_cluster_;
  bool is_low_rank_;
  eigenMatrix S_;
};  // namespace FMCA

}  // namespace FMCA
#endif

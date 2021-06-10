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
template <typename H2ClusterTree> class H2Matrix {
public:
  typedef typename H2ClusterTree::value_type value_type;
  typedef typename H2ClusterTree::eigenMatrix eigenMatrix;
  enum Admissibility { Refine = 0, LowRank = 1, Dense = 2 };
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  H2Matrix() {}
  template <typename Functor>
  H2Matrix(const Eigen::Matrix<value_type, H2ClusterTree::dimension,
                               Eigen::Dynamic> &P,
           const H2ClusterTree &CT, const Functor &fun, value_type eta = 0.8) {
    init(P, CT, fun, eta);
  }

  template <typename Functor>
  void init(const Eigen::Matrix<value_type, H2ClusterTree::dimension,
                                Eigen::Dynamic> &P,
            const H2ClusterTree &CT, const Functor &fun, value_type eta = 0.8) {
    computeH2Matrix(P, CT, CT, fun, eta);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
private:
  //////////////////////////////////////////////////////////////////////////////
  template <typename Functor>
  void computeH2Matrix(const Eigen::Matrix<value_type, H2ClusterTree::dimension,
                                           Eigen::Dynamic> &P,
                       const H2ClusterTree &CT1, const H2ClusterTree &CT2,
                       const Functor &fun, value_type eta) {
    Admissibility adm = compareCluster(CT1, CT2, eta);
    if (adm == LowRank) {
      const eigenMatrix &Xi = CT1.TP_interp_->get_Xi();
      S_.resize(Xi.cols(), Xi.cols());
      for (auto j = 0; j < S_.cols(); ++j)
        for (auto i = 0; i < S_.rows(); ++i)
          S_(i, j) =
              fun((CT1.cluster_->get_bb().col(2).array() * Xi.col(i).array() +
                   CT1.cluster_->get_bb().col(0).array())
                      .matrix(),
                  (CT2.cluster_->get_bb().col(2).array() * Xi.col(j).array() +
                   CT2.cluster_->get_bb().col(0).array())
                      .matrix());
    } else if (adm == Refine) {
      sons_.resize(CT1.sons_.size(), CT2.sons_.size());
      for (auto j = 0; j < CT2.sons_.size(); ++j)
        for (auto i = 0; i < CT1.sons_.size(); ++i)
          sons_(i, j).computeH2Matrix(P, CT1.sons_[i], CT2.sons_[j], fun, eta);
    } else {
      S_.resize(CT1.cluster_->get_indices().size(),
                CT2.cluster_->get_indices().size());
      for (auto j = 0; j < CT2.cluster_->get_indices().size(); ++j)
        for (auto i = 0; i < CT1.cluster_->get_indices().size(); ++i)
          S_(i, j) = fun(P.col(CT1.cluster_->get_indices()[i]),
                         P.col(CT2.cluster_->get_indices()[j]));
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  value_type computeDistance(const H2ClusterTree &cluster1,
                             const H2ClusterTree &cluster2) {
    const value_type row_radius =
        0.5 * cluster1.cluster_->get_bb().col(2).norm();
    const value_type col_radius =
        0.5 * cluster2.cluster_->get_bb().col(2).norm();
    const value_type dist = 0.5 * (cluster1.cluster_->get_bb().col(0) -
                                   cluster2.cluster_->get_bb().col(0) +
                                   cluster1.cluster_->get_bb().col(1) -
                                   cluster2.cluster_->get_bb().col(1))
                                      .norm() -
                            row_radius - col_radius;
    return dist > 0 ? dist : 0;
  }
  //////////////////////////////////////////////////////////////////////////////
  Admissibility compareCluster(const H2ClusterTree &cluster1,
                               const H2ClusterTree &cluster2, value_type eta) {
    Admissibility retval;
    const value_type dist = computeDistance(cluster1, cluster2);
    const value_type row_radius =
        0.5 * cluster1.cluster_->get_bb().col(2).norm();
    const value_type col_radius =
        0.5 * cluster2.cluster_->get_bb().col(2).norm();
    const value_type radius = row_radius > col_radius ? row_radius : col_radius;

    if (radius > eta * dist) {
      // check if either cluster is a leaf in that case,
      // compute the full matrix block
      if (!cluster1.sons_.size() || !cluster2.sons_.size())
        return Dense;
      else
        return Refine;
    } else
      return LowRank;
  }
  //////////////////////////////////////////////////////////////////////////////
  GenericMatrix<H2Matrix> sons_;
  const H2ClusterTree *row_cluster_;
  const H2ClusterTree *col_cluster_;
  bool is_low_rank_;
  eigenMatrix S_;
};

} // namespace FMCA
#endif

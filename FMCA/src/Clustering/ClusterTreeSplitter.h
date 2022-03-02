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
#ifndef FMCA_CLUSTERING_CLUSTERTREESPLITTER_H_
#define FMCA_CLUSTERING_CLUSTERTREESPLITTER_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief provides different methods to bisect a given cluster
 **/
namespace ClusterSplitter {

template <typename Derived1, typename Derived2, typename Derived3,
          typename logicType>
struct MetisBisection {
  template <class ClusterTree>
  void operator()(const Eigen::MatrixBase<Derived1> &V,
                  const Eigen::MatrixBase<Derived2> &F,
                  const std::vector<IndexType> &indices,
                  const Eigen::MatrixBase<Derived3> &bb, ClusterTree &c1,
                  ClusterTree &c2, const logicType dual) const {
    // STEP 0: Setup
    Eigen::SparseMatrix<double> A;
    if (dual)
      igl::facet_adjacency_matrix(F, A);
    else
      igl::adjacency_matrix(F, A);
    Eigen::SparseMatrix<double> sub_A;
    const Eigen::Vector<IndexType, Eigen::Dynamic> idx =
        Eigen::Map<const Eigen::Vector<IndexType, Eigen::Dynamic>,
                   Eigen::Unaligned>(indices.data(), indices.size());
    igl::slice(A, idx, idx, sub_A);
    std::vector<std::vector<IndexType>> sub_A_list;
    sub_A_list.resize(indices.size());
    for (int i = 0; i < sub_A.outerSize(); i++) {
      for (typename Eigen::SparseMatrix<double>::InnerIterator it(sub_A, i); it;
           ++it) {
        sub_A_list.at(it.row()).push_back(it.col());
      }
    }
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    int num = 0;
    xadj.push_back(num);
    for (auto item : sub_A_list) {
      for (auto idx : item) {
        adjncy.push_back(idx);
        num++;
      }
      xadj.push_back(num);
    }

    idx_t nvtxs = xadj.size() - 1;
    idx_t nEdges = adjncy.size() / 2;
    idx_t ncon = 1;
    idx_t nParts = 2;
    idx_t objval;
    idx_t part[nvtxs];
    // Step 1: Partition
    int ret =
        METIS_PartGraphKway(&nvtxs, &ncon, &xadj[0], &adjncy[0], NULL, NULL,
                            NULL, &nParts, NULL, NULL, NULL, &objval, part);
    // Step 2: Split
    for (auto i = 0; i < indices.size(); ++i)
      if (part[i] == 0)
        c1.indices_.push_back(indices[i]);
      else
        c2.indices_.push_back(indices[i]);
    c2.indices_begin_ += c1.indices_.size();

    // Step 3: Bounding box
    if (dual) {
      c1.bb_ = bb;
      c1.bb_.col(0) = V.row(F(c1.indices_[0], 0)).transpose();
      c1.bb_.col(1) = c1.bb_.col(0);
      for (auto i = 0; i < c1.indices_.size(); ++i)
        for (auto k = 0; k < F.cols(); ++k) {
          c1.bb_.col(0).array() = c1.bb_.col(0).array().min(
              V.row(F(c1.indices_[i], k)).transpose().array());
          c1.bb_.col(1).array() = c1.bb_.col(1).array().max(
              V.row(F(c1.indices_[i], k)).transpose().array());
        }
      c2.bb_ = bb;
      c2.bb_.col(0) = V.row(F(c2.indices_[0], 0)).transpose();
      c2.bb_.col(1) = c2.bb_.col(0);
      for (auto i = 0; i < c2.indices_.size(); ++i)
        for (auto k = 0; k < F.cols(); ++k) {
          c2.bb_.col(0).array() = c2.bb_.col(0).array().min(
              V.row(F(c2.indices_[i], k)).transpose().array());
          c2.bb_.col(1).array() = c2.bb_.col(1).array().max(
              V.row(F(c2.indices_[i], k)).transpose().array());
        }

    } else {
      c1.bb_ = bb;
      c1.bb_.col(0) = V(c1.indices_, Eigen::all).colwise().minCoeff();
      c1.bb_.col(1) = V(c1.indices_, Eigen::all).colwise().maxCoeff();
      c2.bb_ = bb;
      c2.bb_.col(0) = V(c2.indices_, Eigen::all).colwise().minCoeff();
      c2.bb_.col(1) = V(c2.indices_, Eigen::all).colwise().maxCoeff();
    }
    c1.bb_.col(2) = c1.bb_.col(1) - c1.bb_.col(0);
    c2.bb_.col(2) = c2.bb_.col(1) - c2.bb_.col(0);
  }
};

template <typename ValueType>
struct GeometricBisection {
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  template <class ClusterTree>
  void operator()(const eigenMatrix &P, const std::vector<IndexType> &indices,
                  const eigenMatrix &bb, ClusterTree &c1,
                  ClusterTree &c2) const {
    // assign bounding boxes by longest edge bisection
    IndexType longest;
    bb.col(2).maxCoeff(&longest);
    c1.bb_ = bb;
    c1.bb_(longest, 2) *= 0.5;
    c1.bb_(longest, 1) -= c1.bb_(longest, 2);
    c2.bb_ = bb;
    c2.bb_(longest, 2) *= 0.5;
    c2.bb_(longest, 0) += c2.bb_(longest, 2);
    // now split the index vector
    for (auto i = 0; i < indices.size(); ++i)
      if ((P.col(indices[i]).array() <= c1.bb_.col(1).array()).all() &&
          (P.col(indices[i]).array() >= c1.bb_.col(0).array()).all())
        c1.indices_.push_back(indices[i]);
      else
        c2.indices_.push_back(indices[i]);
    c2.indices_begin_ += c1.indices_.size();
  }
};

template <typename Derived>
struct CoordinateCompare {
  const typename Eigen::MatrixBase<Derived> &P_;
  Eigen::Index cmp_;
  CoordinateCompare(const Eigen::MatrixBase<Derived> &P, Eigen::Index cmp)
      : P_(P), cmp_(cmp){};

  bool operator()(IndexType i, IndexType &j) {
    return P_(cmp_, i) < P_(cmp_, j);
  }
};

template <typename ValueType>
struct CardinalityBisection {
  typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  template <class ClusterTree>
  void operator()(const eigenMatrix &P, const std::vector<IndexType> &indices,
                  const eigenMatrix &bb, ClusterTree &c1,
                  ClusterTree &c2) const {
    std::vector<IndexType> sorted_indices;
    IndexType longest;
    // assign bounding boxes by longest edge division
    bb.col(2).maxCoeff(&longest);
    sorted_indices = indices;
    // sort father index set with respect to the longest edge component
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              CoordinateCompare<eigenMatrix>(P, longest));
    c1.indices_ = std::vector<IndexType>(
        sorted_indices.begin(),
        sorted_indices.begin() + sorted_indices.size() / 2);
    c2.indices_ = std::vector<IndexType>(
        sorted_indices.begin() + sorted_indices.size() / 2,
        sorted_indices.end());
    c2.indices_begin_ += c1.indices_.size();
    c1.bb_ = bb;
    c1.bb_(longest, 1) = P(longest, c1.indices_.back());
    c1.bb_(longest, 2) = c1.bb_(longest, 1) - c1.bb_(longest, 0);
    c2.bb_ = bb;
    c2.bb_(longest, 0) = P(longest, c2.indices_.front());
    c2.bb_(longest, 2) = c2.bb_(longest, 1) - c2.bb_(longest, 0);
  }
};

}  // namespace ClusterSplitter
}  // namespace FMCA
#endif

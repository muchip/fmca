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
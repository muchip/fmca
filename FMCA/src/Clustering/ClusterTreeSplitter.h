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
#ifndef FMCA_CLUSTERING_CLUSTERTREESPLITTER_H_
#define FMCA_CLUSTERING_CLUSTERTREESPLITTER_H_

namespace FMCA {

/**
 *  \ingroup Clustering
 *  \brief provides different methods to bisect a given cluster
 **/
namespace ClusterSplitter {

struct GeometricBisection {
  static std::string splitterName() { return "GeometricBisection"; }
  template <class ClusterTree>
  void operator()(const Matrix &P, const std::vector<Index> &indices,
                  const Matrix &bb, ClusterTree &c1, ClusterTree &c2) const {
    // assign bounding boxes by longest edge bisection
    Index longest;
    bb.col(2).maxCoeff(&longest);
    c1.bb_ = bb;
    c1.bb_(longest, 2) *= 0.5;
    c1.bb_(longest, 1) -= c1.bb_(longest, 2);
    c2.bb_ = bb;
    c2.bb_(longest, 2) = c1.bb_(longest, 2);
    c2.bb_(longest, 0) = c1.bb_(longest, 1);
    // now split the index vector
    for (auto i = 0; i < indices.size(); ++i)
      if (P(longest, indices[i]) <= c1.bb_(longest, 1))
        c1.indices_.push_back(indices[i]);
      else
        c2.indices_.push_back(indices[i]);
    c2.indices_begin_ += c1.indices_.size();
    assert(c1.indices_.size() + c2.indices_.size() == indices.size() &&
           "lost indices");
  }
};

struct CoordinateCompare {
  const Matrix &P_;
  Eigen::Index cmp_;
  CoordinateCompare(const Matrix &P, Index cmp) : P_(P), cmp_(cmp){};

  bool operator()(Index i, Index &j) { return P_(cmp_, i) < P_(cmp_, j); }
};

struct CardinalityBisection {
  static std::string splitterName() { return "CardinalityBisection"; }
  template <class ClusterTree>
  void operator()(const Matrix &P, const std::vector<Index> &indices,
                  const Matrix &bb, ClusterTree &c1, ClusterTree &c2) const {
    std::vector<Index> sorted_indices;
    Index longest;
    // assign bounding boxes by longest edge division
    bb.col(2).maxCoeff(&longest);
    sorted_indices = indices;
    // sort father index set with respect to the longest edge component
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              CoordinateCompare(P, longest));
    c1.indices_ =
        std::vector<Index>(sorted_indices.begin(),
                           sorted_indices.begin() + sorted_indices.size() / 2);
    c2.indices_ =
        std::vector<Index>(sorted_indices.begin() + sorted_indices.size() / 2,
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

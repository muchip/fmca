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
  template <class CTNode>
  void operator()(const Matrix &P, CTNode &c1, CTNode &c2) const {
    // assign bounding boxes by longest edge bisection
    Index longest = 0;
    Index *idcs = c1.indices_.get();
    Scalar pivot = 0;
    c1.bb_.col(2).maxCoeff(&longest);
    c1.bb_(longest, 2) *= 0.5;
    c1.bb_(longest, 1) -= c1.bb_(longest, 2);
    c2.bb_(longest, 2) = c1.bb_(longest, 2);
    c2.bb_(longest, 0) = c1.bb_(longest, 1);
    // now split the index vector
    pivot = c1.bb_(longest, 1);
    Index low = 0;
    Index high = c1.block_size_ - 1;
    Index swp = 0;
    Index offs = c1.indices_begin_;
    while (low < high) {
      while (low < high && P(longest, idcs[offs + low]) <= pivot) ++low;
      while (P(longest, idcs[offs + high]) > pivot) --high;
      if (low < high) std::swap(idcs[offs + low], idcs[offs + high]);
    }
    c1.block_size_ = high;
    c2.block_size_ -= high;
    c2.indices_begin_ += high;
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
  template <class CTNode>
  void operator()(const Matrix &P, CTNode &c1, CTNode &c2) const {
    // assign bounding boxes by longest edge division
    Index longest = 0;
    Index *idcs = c1.indices_.get();
    c1.bb_.col(2).maxCoeff(&longest);
    // sort father index set with respect to the longest edge component
    std::sort(idcs + c1.indices_begin_,
              idcs + c1.indices_begin_ + c1.block_size_,
              CoordinateCompare(P, longest));
    c1.block_size_ /= 2;
    c2.block_size_ -= c1.block_size_;
    c2.indices_begin_ += c1.block_size_;
    c1.bb_(longest, 1) =
        P(longest, idcs[c1.indices_begin_ + c1.block_size_ - 1]);
    c1.bb_(longest, 2) = c1.bb_(longest, 1) - c1.bb_(longest, 0);
    c2.bb_(longest, 0) = P(longest, idcs[c2.indices_begin_]);
    c2.bb_(longest, 2) = c2.bb_(longest, 1) - c2.bb_(longest, 0);
  }
};

}  // namespace ClusterSplitter
}  // namespace FMCA
#endif

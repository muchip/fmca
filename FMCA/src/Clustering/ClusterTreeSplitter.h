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
    Index longest = 0;
    c1.bb_.col(2).maxCoeff(&longest);
    c1.bb_(longest, 2) *= 0.5;
    c1.bb_(longest, 1) -= c1.bb_(longest, 2);
    c2.bb_(longest, 2) = c1.bb_(longest, 2);
    c2.bb_(longest, 0) = c1.bb_(longest, 1);
    const Scalar pivot = c1.bb_(longest, 1);
    Index *first = c1.indices_.get() + c1.indices_begin_;
    Index *last = first + c1.block_size_;
    Index *mid = std::partition(
        first, last, [&](Index i) { return P(longest, i) <= pivot; });
    const Index low = Index(mid - first);
    c1.block_size_ = low;
    c2.block_size_ -= low;
    c2.indices_begin_ += low;
    c1.c_ = Vector::Unit(P.rows(), longest);
    c1.r_ = pivot;
  }
};

struct GeometricKDSplitting {
  static std::string splitterName() { return "GeometricKDSplitting"; }
  template <class ChildVector>
  void operator()(const Matrix &P, ChildVector &children) const {
    const auto &parent = children[0].dad().node();
    Index num_children = children.size();

    // compute the pivot in each dimension
    const Index d = P.rows();
    std::vector<Scalar> pivots(d);
    for (Index i = 0; i < d; ++i) {
      pivots[i] = parent.bb_(i, 0) + parent.bb_(i, 2) * 0.5;
    }

    // initialize the children's bounding boxes and then modify them based on
    // the binary representation
    for (Index child_idx = 0; child_idx < num_children; ++child_idx) {
      children[child_idx].node().bb_ = parent.bb_;
      for (Index dim = 0; dim < d; ++dim) {
        bool upper_half = (child_idx >> dim) & 1;
        children[child_idx].node().bb_(dim, 2) *=
            0.5;  // halve the bb dimension
        if (upper_half) {
          children[child_idx].node().bb_(dim, 0) = pivots[dim];
        } else {
          children[child_idx].node().bb_(dim, 1) = pivots[dim];
        }
      }
    }

    // points cluster assignment, create k temporary arrays
    std::vector<std::vector<Index>> child_indices(num_children);
    Index *parent_idcs = parent.indices_.get();
    Index parent_starting_index = parent.indices_begin_;

    for (Index i = 0; i < parent.block_size_; ++i) {
      Index point_idx = parent_idcs[parent_starting_index + i];
      Index child_idx = 0;
      for (Index dim = 0; dim < d; ++dim) {
        if (P(dim, point_idx) > pivots[dim]) {
          child_idx |= (1 << dim);
        }
      }
      child_indices[child_idx].push_back(point_idx);
    }

    Index current_starting_points = parent.indices_begin_;
    for (Index child_idx = 0; child_idx < num_children; ++child_idx) {
      children[child_idx].node().block_size_ = child_indices[child_idx].size();
      children[child_idx].node().indices_begin_ = current_starting_points;
      // copy indices back to the shared array
      for (Index i = 0; i < children[child_idx].node().block_size_; ++i) {
        parent_idcs[current_starting_points + i] = child_indices[child_idx][i];
      }
      current_starting_points += children[child_idx].node().block_size_;
    }
  }
};

struct CoordinateCompare {
  const Matrix &P_;
  Index cmp_;
  CoordinateCompare(const Matrix &P, Index cmp) : P_(P), cmp_(cmp) {};

  bool operator()(Index i, Index &j) { return P_(cmp_, i) < P_(cmp_, j); }
};

struct CardinalityBisection {
  static std::string splitterName() { return "CardinalityBisection"; }
  template <class CTNode>
  void operator()(const Matrix &P, CTNode &c1, CTNode &c2) const {
    Index longest = 0;
    c1.bb_.col(2).maxCoeff(&longest);
    const CoordinateCompare cmp(P, longest);
    Index *first = c1.indices_.get() + c1.indices_begin_;
    Index *last = first + c1.block_size_;
    const Index n1 = c1.block_size_ / 2;
    Index *mid = first + n1;
    std::nth_element(first, mid, last, cmp);
    c1.block_size_ = n1;
    c2.block_size_ -= n1;
    c2.indices_begin_ += n1;
    c1.bb_(longest, 1) = P(longest, *std::max_element(first, mid, cmp));
    c1.bb_(longest, 2) = c1.bb_(longest, 1) - c1.bb_(longest, 0);
    c2.bb_(longest, 0) = P(longest, *mid);
    c2.bb_(longest, 2) = c2.bb_(longest, 1) - c2.bb_(longest, 0);
    // hyperplane, exact only for distinct coordinates
    c1.c_ = Vector::Unit(P.rows(), longest);
    c1.r_ = c1.bb_(longest, 1);
  }
};

struct RandomProjection {
  static std::string splitterName() { return "RandomProjection"; }
  template <class CTNode>
  void operator()(const Matrix &P, CTNode &c1, CTNode &c2) const {
    Index *idcs = c1.indices_.get() + c1.indices_begin_;
    const Index D = P.rows();
    const Index bsize = c1.block_size_;
    const Index seed = Index(std::random_device{}()) ^ Index(time(0));
    std::mt19937 mt(seed);
    std::normal_distribution<Scalar> dist(0.0, 1.0);
    Vector v(D);
    for (Index i = 0; i < D; ++i) v(i) = dist(mt);
    v.normalize();
    // project all points into the random direction
    Vector projections(bsize);
    if (bsize > 10000) {
#pragma omp parallel for
      for (Index i = 0; i < bsize; ++i) projections(i) = P.col(idcs[i]).dot(v);
    } else {
      for (Index i = 0; i < bsize; ++i) projections(i) = P.col(idcs[i]).dot(v);
    }
    // median split on the projections
    std::vector<Index> local_idcs(bsize);
    std::iota(local_idcs.begin(), local_idcs.end(), 0);
    const Index n1 = bsize / 2;
    std::vector<Index>::iterator nth = local_idcs.begin() + n1;
    std::nth_element(
        local_idcs.begin(), nth, local_idcs.end(),
        [&](Index a, Index b) { return projections(a) < projections(b); });
    std::vector<Index>::const_iterator pmax = std::max_element(
        local_idcs.begin(), nth,
        [&](Index a, Index b) { return projections(a) < projections(b); });
    c1.r_ = projections(*pmax);
    c1.c_ = std::move(v);
    std::vector<Index> new_idcs(bsize);
    for (Index i = 0; i < bsize; ++i) new_idcs[i] = idcs[local_idcs[i]];
    std::copy(new_idcs.begin(), new_idcs.end(), idcs);
    c1.block_size_ = n1;
    c2.block_size_ -= n1;
    c2.indices_begin_ += n1;
  }
};

}  // namespace ClusterSplitter
}  // namespace FMCA
#endif

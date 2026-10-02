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

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
    Index offs = c1.indices_begin_;
    while (low < high) {
      while (low < high && P(longest, idcs[offs + low]) <= pivot) ++low;
      while (high > 0 && P(longest, idcs[offs + high]) > pivot) --high;
      if (low < high) std::swap(idcs[offs + low], idcs[offs + high]);
    }
    c1.block_size_ = low;
    c2.block_size_ -= low;
    c2.indices_begin_ += low;
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

struct ArrayCompare {
  const Vector &v_;
  ArrayCompare(const Vector &v) : v_(v){};
  bool operator()(Index i, Index &j) { return v_(i) < v_(j); }
};

struct RandomProjection {
  static std::string splitterName() { return "RandomProjection"; }
  template <class CTNode>
  void operator()(const Matrix &P, CTNode &c1, CTNode &c2) const {
    Index *idcs = c1.indices_.get() + c1.indices_begin_;
    const Index D = P.rows();
    const Index bsize = c1.block_size_;
    const Scalar sqrtD = std::sqrt(Scalar(D));
    std::vector<Index> loc_idcs(bsize);
    std::iota(loc_idcs.begin(), loc_idcs.end(), 0);
    // create random direction
    Vector v = Matrix::Random(D, 1);
    Vector projections(bsize);
    v *= (1. / v.norm());
    // project all points into the random direction
    for (Index i = 0; i < bsize; ++i) projections(i) = P.col(idcs[i]).dot(v);
    // sort father index set with respect to the projections array
    {
      std::sort(loc_idcs.begin(), loc_idcs.end(), ArrayCompare(projections));
      std::vector<Index> sorted_indices(bsize);
      for (Index i = 0; i < bsize; ++i) sorted_indices[i] = idcs[loc_idcs[i]];
      std::memcpy(idcs, sorted_indices.data(), bsize * sizeof(Index));
    }
    // determine splitting point (median assumes that array is never empty)
    const Scalar median = bsize % 2
                              ? projections(loc_idcs[bsize / 2])
                              : 0.5 * (projections(loc_idcs[bsize / 2]) +
                                       projections(loc_idcs[bsize / 2 - 1]));
    // use middle point along the random projection direction
    const Vector x = P.col(idcs[bsize / 2]);
    Scalar max_dist = 0;
    // determine point of max distance
    for (Index i = 0; i < bsize; ++i) {
      const Scalar dist = (x - P.col(idcs[i])).norm();
      max_dist = max_dist > dist ? max_dist : dist;
    }
    // set delta
    const Scalar delta = 6. *
                         (2. * Scalar(std::rand()) / Scalar(RAND_MAX) - 1) *
                         max_dist / sqrtD;
    const Scalar medpdelta = median + delta;
    // use that we already sorted the array, so we can determine the
    // splitting point using binary search
    Index split_bsize = 0;
    {
      Index count = bsize;
      Index step = 0;
      Index it = 0;
      while (count > 0) {
        it = split_bsize;
        step = count / 2;
        it += step;
        if (projections(loc_idcs[it]) < medpdelta) {
          split_bsize = ++it;
          count -= step + 1;
        } else
          count = step;
      }
    }
    c1.block_size_ = split_bsize;
    c2.block_size_ -= c1.block_size_;
    c2.indices_begin_ += c1.block_size_;
    // note that no bounding boxes are updated here as this does not make
    // sense rather let this be handled by shrinktofit
  }
};

struct FastRandomProjection {
  static std::string splitterName() { return "FastRandomProjection"; }
  template <class CTNode>
  void operator()(const Matrix &P, CTNode &c1, CTNode &c2) const {
    Index *idcs = c1.indices_.get() + c1.indices_begin_;
    const Index D = P.rows();
    const Index bsize = c1.block_size_;
    const Scalar sqrtD = std::sqrt(Scalar(D));
    Scalar split_ratio = 0;
    Index low = 0;
    Index high = 0;

    // create random direction
    Vector projections(bsize);
    do {
      Vector v = Matrix::Random(D, 1);
      v *= (1. / v.norm());
      // project all points into the random direction
      for (Index i = 0; i < bsize; ++i) projections(i) = P.col(idcs[i]).dot(v);
      const Scalar mean = projections.mean();
      // use middle point along the random projection direction
      const Vector x = P.col(idcs[0]);
      Scalar max_dist = 0;
      // determine point of max distance
      for (Index i = 0; i < bsize; ++i) {
        const Scalar dist = (x - P.col(idcs[i])).norm();
        max_dist = max_dist > dist ? max_dist : dist;
      }
      // set delta
      const Scalar rdm = 2. * Scalar(std::rand()) / Scalar(RAND_MAX) - 1.;
      const Scalar delta = rdm * 6. * max_dist / sqrtD;
      const Scalar pivot = mean + delta;
      // now split the index vector
      low = 0;
      high = bsize - 1;
      while (low < high) {
        while (low < high && projections(low) <= pivot) ++low;
        while (high > 0 && projections(high) > pivot) --high;
        if (low < high) {
          std::swap(idcs[low], idcs[high]);
          std::swap(projections[low], projections[high]);
        }
      }
      split_ratio = low > bsize - low ? Scalar(bsize - low) / Scalar(low)
                                      : Scalar(low) / Scalar(bsize - low);
    } while (split_ratio < 0.01);
    c1.block_size_ = low;
    c2.block_size_ -= low;
    c2.indices_begin_ += low;
    // note that no bounding boxes are updated here as this does not make
    // sense rather let this be handled by shrinktofit
  }
};

}  // namespace ClusterSplitter
}  // namespace FMCA
#endif

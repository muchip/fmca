// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SEMILAGRANGIAN_NEARESTNEIGHBORINTERPOLATOR2D_H_
#define FMCA_SEMILAGRANGIAN_NEARESTNEIGHBORINTERPOLATOR2D_H_

namespace FMCA {

/**
 *  \brief Low-order monotone reconstruction for the semi-Lagrangian
 *         scheme. For each query point we collect its k nearest data
 *         sites, take an inverse-distance-weighted average of the
 *         corresponding values, and clip the result to [min, max] of
 *         those k values. The clip enforces a discrete maximum
 *         principle (the reconstructed value never overshoots its
 *         neighbors), which is the property we want near shocks.
 *
 *  The implementation uses brute-force kNN: O(NM) per evaluation and
 *  O(Nk) memory. This is intentional - the low-order branch is invoked
 *  only on a (small) flagged set of departure points, so a smarter
 *  data structure is not needed here.
 */

class NearestNeighborInterpolator2D {
 public:
  NearestNeighborInterpolator2D() = default;

  void init(const Matrix &P, Index k = 4) {
    P_ = P;
    k_ = k > 0 ? k : 1;
    if (k_ >= static_cast<Index>(P_.cols())) k_ = P_.cols() - 1;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Evaluate the low-order interpolant at the query points @p Q
  /// (DIM x M) using the data values @p u (size N).
  Vector evaluate(const Vector &u, const Matrix &Q) const {
    const Index M = Q.cols();
    const Index N = P_.cols();
    Vector retval(M);
#pragma omp parallel for
    for (Index j = 0; j < M; ++j) {
      // collect the k nearest neighbours by simple insertion sort
      std::vector<std::pair<Scalar, Index>> heap;
      heap.reserve(k_ + 1);
      for (Index i = 0; i < N; ++i) {
        const Scalar d2 = (P_.col(i) - Q.col(j)).squaredNorm();
        if (heap.size() < static_cast<size_t>(k_)) {
          heap.emplace_back(d2, i);
          std::push_heap(heap.begin(), heap.end());
        } else if (d2 < heap.front().first) {
          std::pop_heap(heap.begin(), heap.end());
          heap.back() = std::make_pair(d2, i);
          std::push_heap(heap.begin(), heap.end());
        }
      }
      // inverse-distance weighting + clip to [min, max] of neighbour values
      Scalar wsum = 0;
      Scalar acc = 0;
      Scalar umin = std::numeric_limits<Scalar>::infinity();
      Scalar umax = -std::numeric_limits<Scalar>::infinity();
      for (const auto &p : heap) {
        const Scalar d = std::sqrt(p.first);
        const Scalar w = 1.0 / (d + FMCA_ZERO_TOLERANCE);
        acc += w * u(p.second);
        wsum += w;
        umin = std::min(umin, u(p.second));
        umax = std::max(umax, u(p.second));
      }
      Scalar val = wsum > 0 ? acc / wsum : 0;
      val = std::min(std::max(val, umin), umax);
      retval(j) = val;
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  Index k() const { return k_; }
  const Matrix &points() const { return P_; }

 private:
  Matrix P_;
  Index k_ = 4;
};

}  // namespace FMCA

#endif  // FMCA_SEMILAGRANGIAN_NEARESTNEIGHBORINTERPOLATOR2D_H_

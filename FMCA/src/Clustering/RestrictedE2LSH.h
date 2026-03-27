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
#ifndef FMCA_CLUSTERING_RESTRICTEDE2LSH_H_
#define FMCA_CLUSTERING_RESTRICTEDE2LSH_H_

#include "../util/Macros.h"
#include <functional>
#include <omp.h> //later remove this
#include <random>
#include <set>
#include <string>
#include <unordered_map>

namespace FMCA {

class RestrictedE2LSH {

private:
  Index k_;
  Index L_;
  Scalar r_;
  Index n_;
  std::vector<std::unordered_map<Index, std::vector<Index>>> hash_tables_;
  Matrix points_hashes_;

public:
  RestrictedE2LSH() {}

  void init(const Matrix &P, const Index k, const Index L, const Scalar r,
            const Index seed = 0) {
    // points are stored as columns of P
    k_ = k;
    L_ = L;
    r_ = r;
    n_ = P.cols();
    points_hashes_.resize(n_, L_);
    hash_tables_.resize(L_);
    Index d = P.rows();

    std::vector<Matrix> A(L_);
    std::vector<Vector> B(L_);

    std::mt19937 gen(seed);
    std::normal_distribution<Scalar> gauss(0.0, 1.0);
    std::uniform_real_distribution<Scalar> uniform(0.0, r_);

    for (Index t = 0; t < L_; t++) {

      // A[t] as d x k matrix of Gaussians for random projections
      A[t] = Matrix(d, k_);
      for (Index i = 0; i < d; i++) {
        for (Index j = 0; j < k_; j++) {
          A[t](i, j) = gauss(gen);
        }
      }

      // pick each (of the k_s) b uniformly from [0,r_]
      B[t] = Vector(k_);
      for (Index j = 0; j < k_; j++) {
        B[t](j) = uniform(gen);
      }
    }

    for (Index t = 0; t < L_; t++) {
      Matrix proj = A[t].transpose() * P; // precompute projections

#pragma omp parallel
      {
        std::unordered_map<Index, std::vector<Index>> local;

#pragma omp for nowait
        for (Index i = 0; i < n_; i++) {
          Vector h = ((proj.col(i).array() + B[t].array()) / r_).floor();

          Index key = gethash(h);
          points_hashes_(i, t) = key;
          local[key].push_back(i);
        }

#pragma omp critical
        for (auto &kv : local) {
          auto &bucket = hash_tables_[t][kv.first];
          bucket.insert(bucket.end(), kv.second.begin(), kv.second.end());
        }
      }
    }
  }

  Index gethash(const Vector &v) const {
    Index seed = 0;
    for (Index j = 0; j < v.size(); j++) {
      Index h = static_cast<Index>(v(j));
      seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2); // boost hash combine
    }
    return seed;
  }

  std::vector<Index> computeAENN(const Matrix &P, const Index &q_idx,
                                 const Scalar epsilon) const {

    std::vector<char> visited(n_, 0);
    std::vector<Index> candidates;
    std::vector<Index> out;
    Scalar epsilon_sqrd = epsilon * epsilon;
    // visited[q_idx] = 1;

    for (Index t = 0; t < L_; t++) {
      Index key =
          points_hashes_(q_idx, t); // filter out null hashes...? camn there be?
      auto it = hash_tables_[t].find(key);
      if (it != hash_tables_[t].end()) {

        for (Index j : it->second) {
          if (!visited[j]) {
            visited[j] = 1;
            candidates.push_back(j);
          }
        }
      }
    }

    out.reserve(candidates.size());
#pragma omp parallel
    {
      std::vector<Index> local; // local buffer for thread
#pragma omp for nowait
      for (Index i = 0; i < candidates.size(); ++i) {
        Index j = candidates[i];
        if ((P.col(j) - P.col(q_idx)).squaredNorm() < epsilon_sqrd)
          local.push_back(j);
      }

#pragma omp critical
      out.insert(out.end(), local.begin(), local.end());
    }

    return out;

    // // following E2LSH take the first 3L points, probably better to induce
    // order
    // // in the early set
    // Index num_to_take = std::min<long>(3 * L_, candidates_vec.size());
    // std::vector<Index> candidates_vec(candidates_vec.begin(),
    //                                   candidates_vec.begin() + num_to_take);

  } // namespace FMCA
}; // namespace FMCA
} // namespace FMCA
#endif
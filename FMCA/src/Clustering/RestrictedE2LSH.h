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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <omp.h> // later remove this
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace FMCA {

class RestrictedE2LSH {
  using HashKey = std::uint64_t;

private:
  Index k_;
  Index L_;
  Scalar r_;
  Index n_;
  std::vector<HashKey> points_hashes_;
  std::vector<std::unordered_map<HashKey, std::vector<Index>>> hash_tables_;

  HashKey &pointHash(const Index i, const Index t) {
    return points_hashes_[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(L_) +
                          static_cast<std::size_t>(t)];
  }

  HashKey pointHash(const Index i, const Index t) const {
    return points_hashes_[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(L_) +
                          static_cast<std::size_t>(t)];
  }

  HashKey mixHash(HashKey seed, HashKey value) const {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
  }

  HashKey hashProjectedColumn(const Matrix &proj, const Vector &b,
                              const Index col_idx) const {
    HashKey seed = 0;
    for (Index j = 0; j < k_; ++j) {
      const long long cell =
          static_cast<long long>(std::floor((proj(j, col_idx) + b(j)) / r_));
      const HashKey h = static_cast<HashKey>(std::hash<long long>{}(cell));
      seed = mixHash(seed, h);
    }
    return seed;
  }

public:
  RestrictedE2LSH() : k_(0), L_(0), r_(0), n_(0) {}

  void init(const Matrix &P, const Index k, const Index L, const Scalar r,
            const Index seed = 0) {
    // points are stored as columns of P
    k_ = k;
    L_ = L;
    r_ = r;
    n_ = P.cols();
    const Index d = P.rows();

    points_hashes_.clear();
    points_hashes_.resize(static_cast<std::size_t>(n_) *
                          static_cast<std::size_t>(L_));

    hash_tables_.assign(static_cast<std::size_t>(L_), {});

    std::mt19937 gen(seed);
    std::normal_distribution<Scalar> gauss(0.0, 1.0);
    std::uniform_real_distribution<Scalar> uniform(0.0, r_);

    for (Index t = 0; t < L_; ++t) {
      Matrix A(d, k_);
      Vector B(k_);

      // A is a d x k matrix of Gaussians for random projections
      for (Index i = 0; i < d; ++i) {
        for (Index j = 0; j < k_; ++j) {
          A(i, j) = gauss(gen);
        }
      }

      // pick each of the k offsets uniformly from [0, r_)
      for (Index j = 0; j < k_; ++j) {
        B(j) = uniform(gen);
      }

      Matrix proj = A.transpose() * P; // precompute projections
      auto &table = hash_tables_[static_cast<std::size_t>(t)];
      table.reserve(static_cast<std::size_t>(n_));

#pragma omp parallel
      {
        std::unordered_map<HashKey, std::vector<Index>> local;

#pragma omp for nowait
        for (Index i = 0; i < n_; ++i) {
          const HashKey key = hashProjectedColumn(proj, B, i);
          pointHash(i, t) = key;
          local[key].push_back(i);
        }

#pragma omp critical
        {
          for (auto &kv : local) {
            auto &bucket = table[kv.first];
            bucket.insert(bucket.end(), kv.second.begin(), kv.second.end());
          }
        }
      }
    }
  }

  std::vector<Index> computeAENN(const Matrix &P, const Index &q_idx,
                                 const Scalar epsilon) const {
    std::vector<Index> candidates;
    const Scalar epsilon_sqrd = epsilon * epsilon;
    const auto q = P.col(q_idx);

    std::size_t reserve_guess = 0;
    for (Index t = 0; t < L_; ++t) {
      const HashKey key = pointHash(q_idx, t);
      const auto &table = hash_tables_[static_cast<std::size_t>(t)];
      const auto it = table.find(key);
      if (it != table.end()) {
        reserve_guess += it->second.size();
      }
    }
    candidates.reserve(reserve_guess);

    for (Index t = 0; t < L_; ++t) {
      const HashKey key = pointHash(q_idx, t);
      const auto &table = hash_tables_[static_cast<std::size_t>(t)];
      const auto it = table.find(key);
      if (it != table.end()) {
        candidates.insert(candidates.end(), it->second.begin(),
                          it->second.end());
      }
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());

    std::vector<Index> out;
    out.reserve(candidates.size());

#pragma omp parallel
    {
      std::vector<Index> local;

#pragma omp for nowait
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Index j = candidates[i];
        if ((P.col(j) - q).squaredNorm() <= epsilon_sqrd) {
          local.push_back(j);
        }
      }

#pragma omp critical
      {
        out.insert(out.end(), local.begin(), local.end());
      }
    }

    return out;
  }
};

} // namespace FMCA

#endif
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
#ifndef FMCA_CLUSTERING_E2LSH_H_
#define FMCA_CLUSTERING_E2LSH_H_

#include "../util/Macros.h"
#include <random>
#include <set>
#include <string>
#include <unordered_map>

namespace FMCA {

class E2LSH {
public:
  E2LSH() {}

  void init(const Matrix &P, const Scalar k, const Scalar L, const Scalar r) {
    // points are stored as columns of P
    k_ = k;
    L_ = L;
    r_ = r;

    // create table, hash, (initialize them)
    hash_tables_.resize(L_);
    A_.resize(L_);
    B_.resize(L_);
    d = P.rows();
    std::random_device rd;
    for (Index t = 0; t < L_; t++) {
      std::mt19937 gen(rd() + t); // seed per table
      std::normal_distribution<Scalar> gauss(0.0, 1.0);
      std::uniform_real_distribution<Scalar> uniform(0.0, r_);

      // A_[t] as d x k matrix of Gaussians for random projections
      A_[t] = Matrix(d, k_);
      for (Index i = 0; i < d; i++) {
        for (Index j = 0; j < k_; j++) {
          A_[t](i, j) = gauss(gen);
        }
      }

      // pick each (of the k_s) b uniformly from [0,r_]
      B_[t] = Vector(k_);
      for (Index j = 0; j < k_; j++) {
        B_[t](j) = uniform(gen);
      }
    }

    // hash_tables[t][b].push_back(p);
    Index n = static_cast<Index>(P.cols());
    const int num_threads = omp_get_max_threads();

    for (Index t = 0; t < L_; t++) {
      Matrix projections = A_[t].transpose() * P; // precompute projections

      for (Index i = 0; i < n; i++) {

        Vector hvec =
            ((projections.col(i).array() + B_[t].array()) / r_).floor();
        hash_tables_[t][gethash(hvec)].push_back(i);
      }
    }
  }

  std::string gethash(const Vector &v) {
    // replace this with something faster avoiding strings
    std::string hash = "";
    for (Index j = 0; j < v.size(); j++) {
      hash += std::to_string(v(j)) +
              "_"; // avoid collisions e.g. "12_3_" vs "1_23_"
    }
    return hash;
  }

  void getANN(const Vector &q) {
    // return both the point(index) and distance
    std::set<Index> candidates;
    for (Index t = 0; t < L_; t++) {
      auto it = hash_tables_[t].find(
          gethash(t, q)); // will contain index of candidates
      if (it != hash_tables_[t].end()) {
        candidates.insert(it->second.begin(), it->second.end());
      }
    }

    // find among the candidates the nearest (under l2 norm) to q
    Index best_idx = -1;
    Scalar best_dist = FMCA_INF;

    // add pragma
    for (auto idx : candidates) {
      Scalar dist = (P.col(idx) - q).squaredNorm();
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = idx;
      }
    }

    return best_dist;
  }

  void addPoint() {} // not for current impl.

private:
  Scalar k_;
  Scalar L_;
  Scalar r_;
  std::vector<std::unordered_map<std::string, std::vector<int>>> hash_tables_;
  std::vector<Matrix> A_;
  std::vector<Vector> B_;
};

} // namespace FMCA
#endif
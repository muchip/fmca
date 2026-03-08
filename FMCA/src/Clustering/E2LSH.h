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

private:
  Index k_;
  Index L_;
  Scalar r_;
  std::vector<std::unordered_map<std::string, std::vector<int>>> hash_tables_;
  std::vector<Matrix> A_;
  std::vector<Vector> B_;

public:
  E2LSH() {}

  void init(const Matrix &P, const Index k, const Index L, const Scalar r) {
    // points are stored as columns of P
    k_ = k;
    L_ = L;
    r_ = r;

    // create table, hash, (initialize them)
    hash_tables_.resize(L_);
    A_.resize(L_);
    B_.resize(L_);
    Index d = P.rows();
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

    Index n = static_cast<Index>(P.cols());
    for (Index t = 0; t < L_; t++) {
      Matrix projections = A_[t].transpose() * P; // precompute projections

      std::vector<std::unordered_map<std::string, std::vector<int>>> local_hash(
          omp_get_max_threads());

#pragma omp parallel
      {

        int tid = omp_get_thread_num();

#pragma omp for
        for (Index i = 0; i < n; i++) {
          Vector hvec =
              ((projections.col(i).array() + B_[t].array()) / r_).floor();

          std::string key = gethash(hvec);
          local_hash[tid][key].push_back(i);
        }
      }

      for (auto &thread_map : local_hash) {

        for (auto &kv : thread_map) {
          auto &main_bucket = hash_tables_[t][kv.first];
          main_bucket.insert(main_bucket.end(), kv.second.begin(),
                             kv.second.end());
        }
      }

      // hash_tables_[t][gethash(hvec)].push_back(i);
    }
  }

  std::string gethash(const Vector &v) const {
    // replace this with something faster avoiding strings
    std::string hash = "";
    for (Index j = 0; j < v.size(); j++) {
      hash += std::to_string(v(j)) +
              "_"; // avoid collisions e.g. "12_3_" vs "1_23_"
    }
    return hash;
  }

  std::vector<Index> computeAENN(const Matrix P, const Vector &q,
                                 const Scalar epsilon) const {
    // return both the point(index) and distance
    std::set<Index> candidates;
    for (Index t = 0; t < L_; t++) {

      Vector projections = A_[t].transpose() * q; // precompute projections
      Vector hvec = ((projections.array() + B_[t].array()) / r_).floor();

      auto it = hash_tables_[t].find(
          gethash(hvec)); // will contain index of candidates
      if (it != hash_tables_[t].end()) {
        candidates.insert(it->second.begin(), it->second.end());
      }
    }

    // find among the candidates the one within distance epsilon
    std::set<Index> aenn;
    Scalar epsilon_sqrd = epsilon * epsilon;

    std::vector<std::set<Index>> local_aenn(omp_get_max_threads());

    // convert to vector for pragma usage
    std::vector<Index> candidates_vec(candidates.begin(), candidates.end());

    // // following E2LSH take the first 3L points, probably better to induce
    // order
    // // in the early set
    // Index num_to_take = std::min<long>(3 * L_, candidates_vec.size());
    // std::vector<Index> candidates_vec(candidates_vec.begin(),
    //                                   candidates_vec.begin() + num_to_take);

#pragma omp parallel
    {

      int tid = omp_get_thread_num();

#pragma omp for
      for (Index c = 0; c < candidates_vec.size(); c++) {
        auto idx = candidates_vec[c];
        Scalar dist_sqrd = (P.col(idx) - q).squaredNorm();
        if (dist_sqrd < epsilon_sqrd) {
          local_aenn[tid].insert(idx);
        }
      }
    }

    // take the union
    for (auto &thread_set : local_aenn) {
      aenn.insert(thread_set.begin(), thread_set.end());
    }

    std::vector<Index> results(aenn.begin(), aenn.end());
    return results;
  } // namespace FMCA
}; // namespace FMCA
} // namespace FMCA
#endif
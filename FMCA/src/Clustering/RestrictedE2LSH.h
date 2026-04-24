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
  Index d_;
  Index block_size_;
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

  struct CsvInfo {
    Index nrows = 0;
    Index ncols = 0;
  };

  static std::vector<Scalar> parseCsvLine(const std::string &line) {
    std::vector<Scalar> values;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      if (cell.empty())
        throw std::runtime_error("Encountered empty CSV field");
      values.push_back(static_cast<Scalar>(std::stod(cell)));
    }
    return values;
  }

  static CsvInfo inspectCsv(const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open())
      throw std::runtime_error("Could not open CSV file: " + path);

    CsvInfo info;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      std::vector<Scalar> vals = parseCsvLine(line);
      if (info.nrows == 0) {
        info.ncols = static_cast<Index>(vals.size());
        if (info.ncols <= 0)
          throw std::runtime_error("CSV must have at least one column");
      } else if (static_cast<Index>(vals.size()) != info.ncols) {
        throw std::runtime_error("Inconsistent CSV row width in file: " + path);
      }
      ++info.nrows;
    }
    return info;
  }

  static Index loadCsvBlock(const std::string &path, const Index start_row,
                            const Index count, const Index expected_cols,
                            Matrix &out) {
    std::ifstream in(path);
    if (!in.is_open())
      throw std::runtime_error("Could not open CSV file: " + path);

    std::string line;
    Index current_row = 0;
    Index loaded = 0;

    out.resize(expected_cols, count);

    while (std::getline(in, line)) {
      if (line.empty())
        continue;

      if (current_row < start_row) {
        ++current_row;
        continue;
      }

      if (loaded >= count)
        break;

      std::vector<Scalar> vals = parseCsvLine(line);
      if (static_cast<Index>(vals.size()) != expected_cols)
        throw std::runtime_error(
            "Inconsistent CSV row width while loading block from: " + path);

      for (Index d = 0; d < expected_cols; ++d)
        out(d, loaded) = vals[d];

      ++loaded;
      ++current_row;
    }

    out.conservativeResize(expected_cols, loaded);
    return loaded;
  }

  static Vector loadCsvRowAsColumn(const std::string &path, const Index row_idx,
                                   const Index expected_cols) {
    std::ifstream in(path);
    if (!in.is_open())
      throw std::runtime_error("Could not open CSV file: " + path);

    std::string line;
    Index current_row = 0;

    while (std::getline(in, line)) {
      if (line.empty())
        continue;

      if (current_row == row_idx) {
        std::vector<Scalar> vals = parseCsvLine(line);
        if (static_cast<Index>(vals.size()) != expected_cols)
          throw std::runtime_error(
              "Inconsistent CSV row width while loading point from: " + path);

        Vector out(expected_cols);
        for (Index i = 0; i < expected_cols; ++i)
          out(i) = vals[i];
        return out;
      }
      ++current_row;
    }

    throw std::runtime_error("Requested row index out of range in file: " +
                             path);
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
    d_ = P.rows();

    points_hashes_.clear();
    points_hashes_.resize(static_cast<std::size_t>(n_) *
                          static_cast<std::size_t>(L_));

    hash_tables_.assign(static_cast<std::size_t>(L_), {});

    std::mt19937 gen(seed);
    std::normal_distribution<Scalar> gauss(0.0, 1.0);
    std::uniform_real_distribution<Scalar> uniform(0.0, r_);

    for (Index t = 0; t < L_; ++t) {
      Matrix A(d_, k_);
      Vector B(k_);

      // A is a d x k matrix of Gaussians for random projections
      for (Index i = 0; i < d_; ++i) {
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

  void init(const std::string &P_path, const Index k, const Index L,
            const Scalar r, const Index block_size, const Index seed = 0) {

    if (block_size <= 0)
      throw std::invalid_argument("block_size must be positive");

    CsvInfo info = inspectCsv(P_path);
    n_ = info.nrows;
    k_ = k;
    L_ = L;
    r_ = r;
    d_ = info.ncols;
    block_size_ = block_size;

    points_hashes_.clear();
    points_hashes_.resize(static_cast<std::size_t>(n_) *
                          static_cast<std::size_t>(L_));

    hash_tables_.assign(static_cast<std::size_t>(L_), {});

    std::mt19937 gen(seed);
    std::normal_distribution<Scalar> gauss(0.0, 1.0);
    std::uniform_real_distribution<Scalar> uniform(0.0, r_);

    for (Index t = 0; t < L_; ++t) {
      Matrix A(d_, k_);
      Vector B(k_);

      // A is a d x k matrix of Gaussians for random projections
      for (Index i = 0; i < d_; ++i) {
        for (Index j = 0; j < k_; ++j) {
          A(i, j) = gauss(gen);
        }
      }

      // pick each of the k offsets uniformly from [0, r_)
      for (Index j = 0; j < k_; ++j) {
        B(j) = uniform(gen);
      }

      auto &table = hash_tables_[static_cast<std::size_t>(t)];
      table.reserve(static_cast<std::size_t>(n_));

      //
      for (Index start = 0; start < n_; start += block_size_) {
        Matrix Pblk;
        const Index nb = loadCsvBlock(P_path, start, block_size_, d_, Pblk);
        Matrix proj_blk = A.transpose() * Pblk;

        for (Index i = 0; i < nb; ++i) {
          const Index global_idx = start + i;
          const HashKey key = hashProjectedColumn(proj_blk, B, i);
          pointHash(global_idx, t) = key;
          table[key].push_back(global_idx);
        }
      }
      // precompute projections
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

  std::vector<Index> computeAENN(const std::string &P_path, const Index &q_idx,
                                 const Scalar epsilon) const {

    std::vector<Index> candidates;
    const Scalar epsilon_sqrd = epsilon * epsilon;
    const Vector q = loadCsvRowAsColumn(P_path, q_idx, d_);

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

    // load each block once
    std::unordered_map<Index, std::vector<Index>> block_to_candidates;
    block_to_candidates.reserve(candidates.size());
    for (Index j : candidates) {
      const Index block_id = j / block_size_;
      block_to_candidates[block_id].push_back(j);
    }
    for (const auto &kv : block_to_candidates) {
      const Index block_id = kv.first;
      const std::vector<Index> &idxs = kv.second;

      const Index start = block_id * block_size_;
      Matrix Pblk;
      const Index nb = loadCsvBlock(P_path, start, block_size_, d_, Pblk);

      for (Index j : idxs) {
        const Index local_idx = j - start;
        if (local_idx >= 0 && local_idx < nb) {
          if ((Pblk.col(local_idx) - q).squaredNorm() <= epsilon_sqrd)
            out.push_back(j);
        }
      }
    }
    return out;
    // end
  }
};

} // namespace FMCA

#endif
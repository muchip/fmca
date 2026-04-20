// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer, Michele Palma
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_MODULUSOFCONTINUITY_LSHDISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_LSHDISCRETEMODULUSOFCONTINUITY_H_

#include "../Clustering/RestrictedE2LSH.h"
#include "../util/Macros.h"
#include "DiscreteModulusOfContinuityBase.h"
#include "omp.h"

namespace FMCA {

class LSHDiscreteModulusOfContinuity
    : public DiscreteModulusOfContinuityBase<LSHDiscreteModulusOfContinuity> {
public:
  typedef DiscreteModulusOfContinuityBase<LSHDiscreteModulusOfContinuity> Base;
  LSHDiscreteModulusOfContinuity() {}

  void init(const Matrix &P, const Matrix &f,
            const std::optional<Scalar> TX = std::nullopt,
            const Scalar step_size = 1, const std::string dx_type = "EUCLIDEAN",
            const std::string dy_type = "EUCLIDEAN", const Index L = 5,
            const Index k = 5) {
    setDistanceType(dx_, dx_type);
    setDistanceType(dy_, dy_type);
    Index num_threads_ = omp_get_max_threads();
    bb_.resize(P.rows(), 3);
    bb_.col(0) = P.rowwise().minCoeff();
    bb_.col(1) = P.rowwise().maxCoeff();
    bb_.col(2) = bb_.col(1) - bb_.col(0); // only if EUCLIDEAN is used.
    const Scalar bb_diam = bb_.col(2).norm();
    TX_ = TX.has_value() ? std::min(TX.value(), bb_diam) : bb_diam;
    TX_ = TX_ > 0 ? TX_ : 0;
    if (TX_ <= 0) {
      Base::tgrid_.resize(1, 0);
      Base::omegat_.resize(1, 0);
      return;
    }

    step_size_ = step_size <= TX_ ? step_size : TX_;
    const Index nbins = std::ceil(TX_ / step_size_) + 1;
    Base::tgrid_.resize(nbins);
    Base::omegat_.resize(nbins);
    for (Index i = 0; i < tgrid_.size(); ++i)
      tgrid_[i] = i * step_size_;

    FMCA::RestrictedE2LSH lsh;
    lsh.init(P, k, L, step_size);

#pragma omp parallel
    {
      std::vector<Scalar> local_omegat(nbins, 0);
#pragma omp for schedule(dynamic)
      for (Index q = 0; q < P.cols(); q++) {
        auto neigh = lsh.computeAENN(P, q, TX_);

        for (Index j : neigh) {
          const Scalar xdist = Base::dx_(P.col(q), P.col(j));
          const Scalar ydist = Base::dy_(f.col(q), f.col(j));

          const Index idx =
              std::min(Index(std::ceil(xdist / step_size_)), nbins - 1);

          local_omegat[idx] = std::max(local_omegat[idx], ydist);
        }
      }
#pragma omp critical
      {
        for (Index k = 0; k < omegat_.size(); ++k)
          omegat_[k] = std::max(omegat_[k], local_omegat[k]);
      }
    }

    for (Index k = 1; k < omegat_.size(); ++k)
      omegat_[k] = std::max(omegat_[k - 1], omegat_[k]);
  }

  void init(const std::string &P_path, const std::string &f_path,
            const std::optional<Scalar> TX = std::nullopt,
            const Scalar step_size = 1, const std::string dx_type = "EUCLIDEAN",
            const std::string dy_type = "EUCLIDEAN", const Index L = 5,
            const Index k = 5, const Index block_size = 1024) {

    setDistanceType(dx_, dx_type);
    setDistanceType(dy_, dy_type);

    if (block_size <= 0)
      throw std::invalid_argument("block_size must be positive");

    const CsvInfo pinfo = inspectCsv(P_path);
    const CsvInfo finfo = inspectCsv(f_path);

    if (pinfo.nrows == 0 || finfo.nrows == 0)
      throw std::runtime_error("Input CSV files must not be empty");
    if (pinfo.nrows != finfo.nrows)
      throw std::runtime_error("P and f must have the same number of rows");

    const Index n_samples = pinfo.nrows;
    const Index p_dim = pinfo.ncols;
    const Index f_dim = finfo.ncols;

    // streaming bounding box for P
    Vector pmin(p_dim), pmax(p_dim);
    {
      std::ifstream pin(P_path);
      if (!pin.is_open())
        throw std::runtime_error("Could not open P CSV file: " + P_path);

      std::string line;
      Index row = 0;
      while (std::getline(pin, line)) {
        if (line.empty())
          continue;

        std::vector<Scalar> vals = parseCsvLine(line);
        if (static_cast<Index>(vals.size()) != p_dim)
          throw std::runtime_error("Inconsistent number of columns in P CSV");

        if (row == 0) {
          for (Index i = 0; i < p_dim; ++i) {
            pmin[i] = vals[i];
            pmax[i] = vals[i];
          }
        } else {
          for (Index i = 0; i < p_dim; ++i) {
            pmin[i] = std::min(pmin[i], vals[i]);
            pmax[i] = std::max(pmax[i], vals[i]);
          }
        }
        ++row;
      }
    }

    bb_.resize(p_dim, 3);
    bb_.col(0) = pmin;
    bb_.col(1) = pmax;
    bb_.col(2) = pmax - pmin;

    const Scalar bb_diam = bb_.col(2).norm();
    TX_ = TX.has_value() ? std::min(TX.value(), bb_diam) : bb_diam;
    TX_ = TX_ > 0 ? TX_ : 0;

    if (TX_ <= 0) {
      Base::tgrid_.resize(1, 0);
      Base::omegat_.resize(1, 0);
      return;
    }

    step_size_ = step_size <= TX_ ? step_size : TX_;
    const Index nbins = std::ceil(TX_ / step_size_) + 1;
    Base::tgrid_.resize(nbins);
    Base::omegat_.resize(nbins);

    for (Index i = 0; i < tgrid_.size(); ++i)
      tgrid_[i] = i * step_size_;

    FMCA::RestrictedE2LSH lsh;
    lsh.init(P_path, k, L, step_size, block_size);

#pragma omp parallel
    {
      std::vector<Scalar> local_omegat(nbins, 0);

#pragma omp for schedule(dynamic)
      for (Index q = 0; q < n_samples; ++q) {
        const Vector pq = loadCsvRowAsColumn(P_path, q, p_dim);
        const Vector fq = loadCsvRowAsColumn(f_path, q, f_dim);

        auto neigh = lsh.computeAENN(P_path, q, TX_);

        for (Index j : neigh) {
          const Vector pj = loadCsvRowAsColumn(P_path, j, p_dim);
          const Vector fj = loadCsvRowAsColumn(f_path, j, f_dim);

          const Scalar xdist = Base::dx_(pq, pj);
          const Scalar ydist = Base::dy_(fq, fj);
          const Index idx =
              std::min(Index(std::ceil(xdist / step_size_)), nbins - 1);

          local_omegat[idx] = std::max(local_omegat[idx], ydist);
        }
      }

#pragma omp critical
      {
        for (Index i = 0; i < omegat_.size(); ++i)
          omegat_[i] = std::max(omegat_[i], local_omegat[i]);
      }
    }

    for (Index i = 1; i < omegat_.size(); ++i)
      omegat_[i] = std::max(omegat_[i - 1], omegat_[i]);
  }

private:
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

private:
  using Base::bb_;
  using Base::dx_;
  using Base::dy_;
  using Base::omegat_;
  using Base::setDistanceType;
  using Base::step_size_;
  using Base::tgrid_;
  using Base::TX_;
}; // namespace FMCA

} // namespace FMCA
#endif

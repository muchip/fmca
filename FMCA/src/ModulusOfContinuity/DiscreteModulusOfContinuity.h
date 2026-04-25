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
#ifndef FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITY_H_

#include "../Clustering/E2LSH.h"
#include "../util/Macros.h"
#include "DiscreteModulusOfContinuityBase.h"
#include <optional>

namespace FMCA {

class DiscreteModulusOfContinuity
    : public DiscreteModulusOfContinuityBase<DiscreteModulusOfContinuity> {
public:
  typedef DiscreteModulusOfContinuityBase<DiscreteModulusOfContinuity> Base;

  DiscreteModulusOfContinuity() {}
  // for now, we only use linearly scaled bins for evaluation of the discrete
  // MOC. Similarly to DM25, it might make sense to use quadratically graded
  // grids to improve resolution of omegat at 0.
  void init(const Matrix &P, const Matrix &f,
            const std::optional<Scalar> TX = std::nullopt,
            const std::optional<Scalar> qX = std::nullopt,
            const Index nbins = 100, const std::string dx_type = "EUCLIDEAN",
            const std::string dy_type = "EUCLIDEAN") {
    setDistanceType(dx_, dx_type);
    setDistanceType(dy_, dy_type);

    // only if EUCLIDEAN or TAXICAB is used.
    bb_.resize(P.rows(), 3);
    bb_.col(0) = P.rowwise().minCoeff();
    bb_.col(1) = P.rowwise().maxCoeff();
    bb_.col(2) = bb_.col(1) - bb_.col(0);
    const Scalar bb_diam = Base::dx_(bb_.col(0), bb_.col(1));
    TX_ = TX.has_value() ? TX.value() : bb_diam;
    TX_ = TX_ > 0 ? TX_ : 0;
    if (TX_ <= 0) {
      Base::tgrid_.resize(1, 0);
      Base::omegat_.resize(1, 0);
      return;
    }

    // compute moc for t in [q_x, T_x]
    Scalar qX_;
    if (!qX.has_value()) {
      qX_ = TX_;
#pragma omp parallel
      {
        Scalar local_qX = TX_;

#pragma omp for schedule(dynamic)
        for (FMCA::Index k = 0; k < P.cols(); ++k) {
          for (FMCA::Index l = 0; l < k; ++l) {
            const Scalar xdist = Base::dx_(P.col(k), P.col(l));

            // Use the smallest strictly positive distance.
            // This avoids log(0), and ignores duplicate points.
            if (xdist > 0)
              local_qX = std::min(local_qX, xdist);
          }
        }

#pragma omp critical
        {
          qX_ = std::min(qX_, local_qX);
        }
      }
    } else {
      qX_ = qX.value();
    }

    Base::tgrid_.resize(nbins);
    Base::omegat_.resize(nbins);
    Base::omegat_.assign(nbins, 0);
    const Scalar log_qX = std::log(qX_);
    const Scalar log_TX = std::log(TX_);
    const Scalar log_step = (log_TX - log_qX) / (nbins - 1);

    for (Index i = 0; i < Base::tgrid_.size(); ++i)
      Base::tgrid_[i] = std::exp(log_qX + i * log_step);

    // Avoid tiny floating-point endpoint drift.
    Base::tgrid_[0] = qX_;
    Base::tgrid_[nbins - 1] = TX_;

    // const Scalar quad_scale = TX_ / ((nbins - 1) * (nbins - 1));
    // for (Index i = 0; i < tgrid_.size(); ++i)
    //   tgrid_[i] = quad_scale * i * i;

#pragma omp parallel
    {
      std::vector<Scalar> local_omegat(nbins, 0);
#pragma omp for schedule(dynamic)
      for (FMCA::Index k = 0; k < P.cols(); ++k) {
        for (FMCA::Index l = 0; l < k; ++l) {
          const Scalar xdist = Base::dx_(P.col(k), P.col(l));
          const Scalar ydist = Base::dy_(f.col(k), f.col(l));

          Index idx = 0;

          if (xdist <= qX_) {
            idx = 0;
          } else if (xdist >= TX_) {
            idx = nbins - 1;
          } else {
            idx =
                static_cast<Index>(std::ceil(std::log(xdist / qX_) / log_step));

            if (idx >= nbins)
              idx = nbins - 1;
          }

          // Index idx =
          //     static_cast<Index>(std::ceil(std::sqrt(xdist / quad_scale)));
          // if (idx >= nbins)
          //   idx = nbins - 1;
          // std::min(Index(std::ceil(xdist / step_size_)), nbins - 1);
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
            const std::string dy_type = "EUCLIDEAN",
            const Index block_size = 1024) {
    setDistanceType(dx_, dx_type);
    setDistanceType(dy_, dy_type);

    if (block_size <= 0)
      throw std::invalid_argument("block_size must be positive");

    CsvInfo pinfo = inspectCsv(P_path);
    CsvInfo finfo = inspectCsv(f_path);

    if (pinfo.nrows == 0 || finfo.nrows == 0)
      throw std::runtime_error("Input CSV files must not be empty");

    if (pinfo.nrows != finfo.nrows)
      throw std::runtime_error("P and f must have the same number of rows");

    const Index n_samples = pinfo.nrows;
    const Index p_dim = pinfo.ncols;
    const Index f_dim = finfo.ncols;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> pmin(p_dim), pmax(p_dim);
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
    const Scalar bb_diam = Base::dx_(bb_.col(0), bb_.col(1));
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

    for (Index i = 0; i < nbins; ++i)
      Base::tgrid_[i] = i * step_size_;

    std::vector<Scalar> global_omegat(nbins, Scalar(0));

    const Index nblocks = (n_samples + block_size - 1) / block_size;

    for (Index bi = 0; bi < nblocks; ++bi) {
      Matrix Pi, fi;
      const Index i_start = bi * block_size;
      const Index ni = loadCsvBlock(P_path, i_start, block_size, p_dim, Pi);
      const Index ni_f = loadCsvBlock(f_path, i_start, block_size, f_dim, fi);

      if (ni != ni_f)
        throw std::runtime_error(
            "P/f block size mismatch while reading block i");

      {
        std::vector<Scalar> local_omegat(nbins, Scalar(0));

#pragma omp parallel
        {
          std::vector<Scalar> thread_omegat(nbins, Scalar(0));

#pragma omp for schedule(dynamic)
          for (Index k = 0; k < ni; ++k) {
            for (Index l = 0; l < k; ++l) {
              const Scalar xdist = Base::dx_(Pi.col(k), Pi.col(l));
              if (xdist > TX_)
                continue;

              const Scalar ydist = Base::dy_(fi.col(k), fi.col(l));
              const Index idx =
                  std::min(Index(std::ceil(xdist / step_size_)), nbins - 1);
              thread_omegat[idx] = std::max(thread_omegat[idx], ydist);
            }
          }

#pragma omp critical
          {
            for (Index b = 0; b < nbins; ++b)
              local_omegat[b] = std::max(local_omegat[b], thread_omegat[b]);
          }
        }

        for (Index b = 0; b < nbins; ++b)
          global_omegat[b] = std::max(global_omegat[b], local_omegat[b]);
      }

      for (Index bj = 0; bj < bi; ++bj) {
        Matrix Pj, fj;
        const Index j_start = bj * block_size;
        const Index nj = loadCsvBlock(P_path, j_start, block_size, p_dim, Pj);
        const Index nj_f = loadCsvBlock(f_path, j_start, block_size, f_dim, fj);

        if (nj != nj_f)
          throw std::runtime_error(
              "P/f block size mismatch while reading block j");

        std::vector<Scalar> local_omegat(nbins, Scalar(0));

#pragma omp parallel
        {
          std::vector<Scalar> thread_omegat(nbins, Scalar(0));

#pragma omp for schedule(dynamic)
          for (Index k = 0; k < ni; ++k) {
            for (Index l = 0; l < nj; ++l) {
              const Scalar xdist = Base::dx_(Pi.col(k), Pj.col(l));
              if (xdist > TX_)
                continue;

              const Scalar ydist = Base::dy_(fi.col(k), fj.col(l));
              const Index idx =
                  std::min(Index(std::ceil(xdist / step_size_)), nbins - 1);
              thread_omegat[idx] = std::max(thread_omegat[idx], ydist);
            }
          }

#pragma omp critical
          {
            for (Index b = 0; b < nbins; ++b)
              local_omegat[b] = std::max(local_omegat[b], thread_omegat[b]);
          }
        }

        for (Index b = 0; b < nbins; ++b)
          global_omegat[b] = std::max(global_omegat[b], local_omegat[b]);
      }
    }

    for (Index b = 0; b < nbins; ++b)
      Base::omegat_[b] = global_omegat[b];

    for (Index b = 1; b < Base::omegat_.size(); ++b)
      Base::omegat_[b] = std::max(Base::omegat_[b - 1], Base::omegat_[b]);
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
      } else {
        if (static_cast<Index>(vals.size()) != info.ncols)
          throw std::runtime_error("Inconsistent CSV row width in file: " +
                                   path);
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

private:
  using Base::bb_;
  using Base::dx_;
  using Base::dy_;
  using Base::omegat_;
  using Base::setDistanceType;
  using Base::step_size_;
  using Base::tgrid_;
};

} // namespace FMCA
#endif

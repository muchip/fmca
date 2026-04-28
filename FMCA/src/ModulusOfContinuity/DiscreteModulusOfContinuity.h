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

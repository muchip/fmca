// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_COVARIANCEKERNEL_SUMKERNEL_H_
#define FMCA_COVARIANCEKERNEL_SUMKERNEL_H_

#include <vector>

namespace FMCA {

class SumKernel : public CovarianceKernel {
 public:
  // kernel_type: one of the CovarianceKernel families ("matern32", ...).
  // scales[m] / weights[m]: length scale sigma_m and weight w_m of component m.
  SumKernel(const std::string &kernel_type, const std::vector<Scalar> &scales,
            const std::vector<Scalar> &weights)
      : weights_(weights) {
    assert(scales.size() == weights.size() &&
           "SumKernel: scales and weights must have the same size");
    components_.reserve(scales.size());  // keep component addresses stable
    for (Scalar s : scales) components_.emplace_back(kernel_type, s);
    setRadialFunction([this](Scalar r) {
      Scalar val = 0.;
      for (std::size_t m = 0; m < components_.size(); ++m)
        val += weights_[m] * components_[m].kernel()(r);
      return val;
    });
  }

  SumKernel(const SumKernel &) = delete;
  SumKernel(SumKernel &&) = delete;
  SumKernel &operator=(const SumKernel &) = delete;
  SumKernel &operator=(SumKernel &&) = delete;

 private:
  std::vector<CovarianceKernel> components_;
  std::vector<Scalar> weights_;
};

}  // namespace FMCA

#endif  // FMCA_COVARIANCEKERNEL_SUMKERNEL_H_

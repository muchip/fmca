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
#ifndef FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITY_H_

namespace FMCA {

class DiscreteModulusOfContinuity {
 public:
  template <typename Derived>
  void init(const ClusterTreeBase<Derived> &ct, const FMCA::Matrix &P,
            const Scalar r, const Index R = 2, const Scalar TX) {
    K_ = std::ceil()
  }

 private:
  std::vector<std::vector<Index>> XNk_indices_;
  std::vector<Scalar> omegaNk_;
  Scalar TX_;
  Scalar r_;
  Index R_;
  Index K_;
};
}  // namespace FMCA
#endif

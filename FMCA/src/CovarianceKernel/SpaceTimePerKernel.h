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
#ifndef FMCA_COVARIANCEKERNEL_SPTPERIODIC_H_
#define FMCA_COVARIANCEKERNEL_SPTPERIODIC_H_

namespace FMCA {
class SpaceTimePerKernel {
 public:
  SpaceTimePerKernel(){};
  SpaceTimePerKernel(FMCA::Scalar l = 1., FMCA::Scalar c = 1.) {
    l_ = l;
    c_ = c;
    ktype_ = "SPTPERIODIC";
    return;
  }

  template <typename derived, typename otherDerived>
  FMCA::Scalar operator()(const Eigen::MatrixBase<derived> &x,
                          const Eigen::MatrixBase<otherDerived> &y) const {
    FMCA::Scalar sr = (x - y).head(2).norm();
    FMCA::Scalar st = std::sin(FMCA_PI * (x - y).tail(1).norm());
    return (1. + sqrt(3) / l_ * sr) * exp(-sqrt(3) * sr / l_) *
           exp(-2. * st * st / (c_ * c_));
  }

  FMCA::Matrix eval(const FMCA::Matrix &PR, const FMCA::Matrix &PC) const {
    FMCA::Matrix retval(PR.cols(), PC.cols());
    for (auto j = 0; j < PC.cols(); ++j)
      for (auto i = 0; i < PR.cols(); ++i)
        retval(i, j) = operator()(PR.col(i), PC.col(j));
    return retval;
  }

  std::string kernelType() const { return ktype_; }

 private:
  FMCA::Scalar l_;
  FMCA::Scalar c_;
  std::string ktype_;
};
}  // namespace FMCA
#endif

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
#ifndef FMCA_COVARIANCEKERNEL_COVARIANCEKERNEL_H_
#define FMCA_COVARIANCEKERNEL_COVARIANCEKERNEL_H_

namespace FMCA {
class CovarianceKernel {
 public:
  CovarianceKernel(){};
  CovarianceKernel(const std::string &ktype, FMCA::Scalar l)
      : ktype_(ktype), l_(l) {
    // transform string to upper and check if kernel is implemented
    for (auto &c : ktype_) c = (char)toupper(c);
    ////////////////////////////////////////////////////////////////////////////
    if (ktype_ == "EXPONENTIAL")
      kernel_ = [this](FMCA::Scalar r) { return exp(-r / l_); };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN32")
      kernel_ = [this](FMCA::Scalar r) {
        return (1. + sqrt(3) / l_ * r) * exp(-sqrt(3) * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN52")
      kernel_ = [this](FMCA::Scalar r) {
        return (1. + sqrt(5) / l_ * r + 5 * r * r / (3 * l_ * l_)) *
               exp(-sqrt(5) * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN72")
      kernel_ = [this](FMCA::Scalar r) {
        return (1. + sqrt(7) / l_ * r + 14. / 5 * pow(r / l_, 2.) +
                7. * sqrt(7) / 15. * pow(r / l_, 3.)) *
               exp(-sqrt(7) * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN92")
      kernel_ = [this](FMCA::Scalar r) {
        return (1. + 3. / l_ * r + 27. / 7. * pow(r / l_, 2.) +
                18. / 7. * pow(r / l_, 3.) + 27. / 35. * pow(r / l_, 4.)) *
               exp(-3 * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "GAUSSIAN")
      kernel_ = [this](FMCA::Scalar r) {
        return exp(-0.5 * r * r / (l_ * l_));
      };
    else
      assert(false && "desired kernel not implemented");
  }

  template <typename derived, typename otherDerived>
  FMCA::Scalar operator()(const Eigen::MatrixBase<derived> &x,
                          const Eigen::MatrixBase<otherDerived> &y) const {
    return kernel_((x - y).norm());
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
  // member variables
  std::function<FMCA::Scalar(FMCA::Scalar)> kernel_;
  std::string ktype_;
  FMCA::Scalar l_;
};
}  // namespace FMCA
#endif

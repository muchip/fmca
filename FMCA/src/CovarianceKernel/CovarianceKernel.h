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
#include <cmath>

namespace FMCA {
class CovarianceKernel {
 public:
  CovarianceKernel() {};
  CovarianceKernel(const CovarianceKernel &other) {
    kernel_ = other.kernel_;
    ktype_ = other.ktype_;
    l_ = other.l_;
    c_ = other.c_;
    nu_ = other.nu_;
  }

  CovarianceKernel(CovarianceKernel &&other) {
    kernel_ = other.kernel_;
    ktype_ = other.ktype_;
    l_ = other.l_;
    c_ = other.c_;
    nu_ = other.nu_;
  }

  CovarianceKernel &operator=(CovarianceKernel other) {
    std::swap(kernel_, other.kernel_);
    std::swap(ktype_, other.ktype_);
    std::swap(l_, other.l_);
    std::swap(c_, other.c_);
    std::swap(nu_, other.nu_);
    return *this;
  }

  CovarianceKernel(const std::string &ktype, Scalar l = 1., Scalar c = 1.,
                   Scalar nu = 1.)
      : ktype_(ktype), l_(l), c_(c), nu_(nu) {
    // transform string to upper and check if kernel is implemented
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    ////////////////////////////////////////////////////////////////////////////
    if (ktype_ == "BIHARMONIC2D")
      kernel_ = [this](Scalar r) {
        return r < FMCA_ZERO_TOLERANCE ? 0 : log(r / l_) * (r / l_) * (r / l_);
      };
    else if (ktype_ == "BIHARMONIC3D")
      kernel_ = [this](Scalar r) { return r / l_; };
    else if (ktype_ == "TRIHARMONIC3D")
      kernel_ = [this](Scalar r) { return (r / l_) * (r / l_) * (r / l_); };
    else if (ktype_ == "EXPONENTIAL")
      kernel_ = [this](Scalar r) { return std::exp(-r / l_); };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN32")
      kernel_ = [this](Scalar r) {
        return (1. + std::sqrt(3) / l_ * r) * std::exp(-std::sqrt(3) * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN52")
      kernel_ = [this](Scalar r) {
        return (1. + std::sqrt(5) / l_ * r + 5 * r * r / (3 * l_ * l_)) *
               std::exp(-std::sqrt(5) * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN72")
      kernel_ = [this](Scalar r) {
        return (1. + std::sqrt(7) / l_ * r + 14. / 5 * std::pow(r / l_, 2.) +
                7. * std::sqrt(7) / 15. * std::pow(r / l_, 3.)) *
               std::exp(-std::sqrt(7) * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN92")
      kernel_ = [this](Scalar r) {
        return (1. + 3. / l_ * r + 27. / 7. * std::pow(r / l_, 2.) +
                18. / 7. * std::pow(r / l_, 3.) +
                27. / 35. * std::pow(r / l_, 4.)) *
               std::exp(-3 * r / l_);
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "GAUSSIAN")
      kernel_ = [this](Scalar r) { return std::exp(-0.5 * r * r / (l_ * l_)); };
    ////////////////////////////////////////////////////////////////////////////

    // else if (ktype_ == "MATERNNU")
    //   kernel_ = [this](Scalar r) {
    //     const Scalar arg = std::sqrt(2 * nu_) * r / l_;
    //     return arg > FMCA_ZERO_TOLERANCE
    //                ? 2. * std::pow(0.5 * arg, nu_) / std::tgamma(nu_) *
    //                      std::cyl_bessel_k(nu_, arg)
    //                : 1.;
    //   };

    /////////////////// compact support
    else if (ktype_ == "WENDLAND20")
      kernel_ = [this](Scalar r) {
        return (r / l_) >= 1. ? 0. : std::pow(1. - (r / l_), 2);
      };
    else if (ktype_ == "WENDLAND21")
      kernel_ = [this](Scalar r) {
        return (r / l_) >= 1.
                   ? 0.
                   : std::pow(1. - (r / l_), 4) * (1. + 4. * (r / l_)) / 20;
      };
    ////////////////////

    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "INVMULTIQUADRIC")
      kernel_ = [this](Scalar r) {
        return 1. / std::sqrt((r / l_) * (r / l_) + c_ * c_);
      };
    else if (ktype_ == "MULTIQUADRIC")
      kernel_ = [this](Scalar r) {
        return std::sqrt((r / l_) * (r / l_) + c_ * c_);
      };
    else
      assert(false && "desired kernel not implemented");
  }

  template <typename derived, typename otherDerived>
  Scalar operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return kernel_((x - y).norm());
  }

  Matrix eval(const Matrix &PR, const Matrix &PC) const {
    Matrix retval(PR.cols(), PC.cols());
    for (auto j = 0; j < PC.cols(); ++j)
      for (auto i = 0; i < PR.cols(); ++i)
        retval(i, j) = operator()(PR.col(i), PC.col(j));
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  const Scalar &nu() const { return nu_; }
  const Scalar &l() const { return l_; }
  const Scalar &c() const { return c_; }
  Scalar &nu() { return nu_; }
  Scalar &l() { return l_; }
  Scalar &c() { return c_; }

  std::string kernelType() const { return ktype_; }
  const std::function<Scalar(Scalar)> &kernel() { return kernel_; }

 private:
  // member variables
  std::function<Scalar(Scalar)> kernel_;
  std::string ktype_;
  Scalar l_;
  Scalar c_;
  Scalar nu_;
};
}  // namespace FMCA
#endif

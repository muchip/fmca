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
#ifndef FMCA_COVARIANCEKERNEL_GRADKERNEL_H_
#define FMCA_COVARIANCEKERNEL_GRADKERNEL_H_
#include <cmath>

namespace FMCA {

class GradKernel {
 public:
  GradKernel() {};
  GradKernel(const GradKernel& other) {
    gradkernel_ = other.gradkernel_;
    ktype_ = other.ktype_;
    l_ = other.l_;
    c_ = other.c_;
    d_ = other.d_;
  }

  GradKernel(GradKernel&& other) noexcept {
    gradkernel_ = std::move(other.gradkernel_);
    ktype_ = std::move(other.ktype_);
    l_ = other.l_;
    c_ = other.c_;
    d_ = other.d_;
  }

  GradKernel& operator=(GradKernel other) {
    std::swap(gradkernel_, other.gradkernel_);
    std::swap(ktype_, other.ktype_);
    std::swap(l_, other.l_);
    std::swap(c_, other.c_);
    std::swap(d_, other.d_);
    return *this;
  }

  GradKernel(const std::string& ktype, Scalar l = 1., Scalar c = 1., int d = 0)
      : ktype_(ktype), l_(l), c_(c), d_(d) {
    // Transform string to upper case
    for (auto& chr : ktype_) chr = (char)toupper(chr);
    ////////////////////////////////////////////////////////////////////////////
    if (ktype_ == "GAUSSIAN_SECOND_DERIVATIVE")
      gradkernel_ = [this](Scalar x, Scalar y, Scalar r, Scalar) {
        return ((x - y) * (x - y) - (l_ * l_)) / (l_ * l_ * l_ * l_) *
               exp(-0.5 * r * r / (l_ * l_));
      };
    else if (ktype_ == "GAUSSIAN_FOURTH_DERIVATIVE")
      gradkernel_ = [this](Scalar x, Scalar y, Scalar r, Scalar) {
        return (3 * pow(l_, 4) - 6 * pow((x - y), 2) * pow(l_, 2) +
                pow((x - y), 4)) /
               pow(l_, 8) * exp(-0.5 * r * r / (l_ * l_));
      };
    else if (ktype_ == "MULTIQUADRIC_SECOND_DERIVATIVE")
      gradkernel_ = [this](Scalar x, Scalar y, Scalar r, Scalar) {
        return (1 / (l_ * l_ * sqrt((r / l_) * (r / l_) + c_ * c_))) *
               (1 - ((x - y) * (x - y) / ((r / l_) * (r / l_) + c_ * c_)));
      };
    else if (ktype_ == "MATERN52_SECOND_DERIVATIVE")
      gradkernel_ = [this](Scalar x, Scalar y, Scalar r, Scalar) {
        return -5 / (3 * l_ * l_) * exp(-sqrt(5) / l_ * r) *
               (1 + sqrt(5) / l_ * r - 5 / (l_ * l_) * (x - y) * (x - y));
      };
    else
      assert(false && "desired gradient kernel not implemented");
  }

  template <typename Derived, typename OtherDerived>
  Scalar operator()(const Eigen::MatrixBase<Derived>& x,
                    const Eigen::MatrixBase<OtherDerived>& y) const {
    return gradkernel_(x[d_], y[d_], (x - y).norm(), x.norm());
  }

  Matrix eval(const Matrix& PR, const Matrix& PC) const {
    Matrix retval(PR.cols(), PC.cols());
    for (int j = 0; j < PC.cols(); ++j)
      for (int i = 0; i < PR.cols(); ++i)
        retval(i, j) = operator()(PR.col(i), PC.col(j));
    return retval;
  }

  std::string gradkernelType() const { return ktype_; }

 private:
  std::string ktype_;
  Scalar l_;
  Scalar c_;
  int d_;
  std::function<Scalar(Scalar, Scalar, Scalar, Scalar)> gradkernel_;
};

}  // namespace FMCA

#endif  // FMCA_GRADKERNEL_GRADKERNEL_H_

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

#ifndef FMCA_LAPLACIANKERNEL_LAPLACIANKERNEL_H_
#define FMCA_LAPLACIANKERNEL_LAPLACIANKERNEL_H_

namespace FMCA {
class LaplacianKernel {
 public:
  LaplacianKernel(){};
  LaplacianKernel(const LaplacianKernel &other) {
    laplaciankernel_ = other.laplaciankernel_;
    ktype_ = other.ktype_;
    l_ = other.l_;
    c_ = other.c_;
    d_ = other.d_;
  }

  LaplacianKernel(LaplacianKernel &&other) {
    laplaciankernel_ = other.laplaciankernel_;
    ktype_ = other.ktype_;
    l_ = other.l_;
    c_ = other.c_;
    d_ = other.d_;
  }

  LaplacianKernel &operator=(LaplacianKernel other) {
    std::swap(laplaciankernel_, other.laplaciankernel_);
    std::swap(ktype_, other.ktype_);
    std::swap(l_, other.l_);
    std::swap(c_, other.c_);
    std::swap(d_, other.d_);
    return *this;
  }

  LaplacianKernel(const std::string &ktype, FMCA::Scalar l = 1., FMCA::Scalar c = 1.,
             int d = 0)
      : ktype_(ktype), l_(l), c_(c), d_(d) {
    // Transform string to upper case
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    ////////////////////////////////////////////////////////////////////////////
    if (ktype_ == "GAUSSIAN")
      laplaciankernel_ = [this](FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar r) {
        return  ((x - y) * (x - y) - (l_ * l_)) / (l_ * l_ * l_ * l_) * exp(-0.5 * r * r / (l_ * l_));
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MULTIQUADRIC")
      laplaciankernel_ = [this](FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar r) {
        return ((l_ * l_) * ((r / l_) * (r / l_) + c_ * c_) - (x - y) * (x - y)) / ( (l_ * l_ * l_ * l_) * pow((r / l_) * (r / l_) + c_ * c_ , 3/2) );
      };
    ////////////////////////////////////////////////////////////////////////////
    else if (ktype_ == "MATERN52")
    laplaciankernel_ = [this](FMCA::Scalar x, FMCA::Scalar y, FMCA::Scalar r) {
      return -5 / (3 * l_ * l_) * exp(-sqrt(5) / l_ * r) * (1 + sqrt(5) / l_ * r - 5 / (l_ * l_) * (x - y) * (x - y));
    };
    else
      assert(false && "desired laplacian kernel not implemented");
  }

  template <typename derived, typename otherDerived>

  FMCA::Scalar operator()(const Eigen::MatrixBase<derived> &x,
                          const Eigen::MatrixBase<otherDerived> &y) const {
    return laplaciankernel_(x[d_], y[d_], (x - y).norm());
  }
  FMCA::Matrix eval(const FMCA::Matrix &PR, const FMCA::Matrix &PC) const {
    FMCA::Matrix retval(PR.cols(), PC.cols());
    for (auto j = 0; j < PC.cols(); ++j)
      for (auto i = 0; i < PR.cols(); ++i)
        retval(i, j) = operator()(PR.col(i), PC.col(j));
    return retval;
  }
  std::string laplaciankernelType() const { return ktype_; }

 private:
  std::function<FMCA::Scalar(FMCA::Scalar, FMCA::Scalar, FMCA::Scalar)>
      laplaciankernel_;
  std::string ktype_;
  FMCA::Scalar l_;
  FMCA::Scalar c_;
  int d_;
};

}  // namespace FMCA

#endif

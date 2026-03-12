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
#ifndef FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITYBASE_H_
#define FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITYBASE_H_

namespace FMCA {

template <typename Derived>
class DiscreteModulusOfContinuityBase {
 public:
  DiscreteModulusOfContinuityBase() {}
  //////////////////////////////////////////////////////////////////////////////
  // return a reference to the derived object
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

  //////////////////////////////////////////////////////////////////////////////
  // exposed the trees init routine
  template <typename... Ts>
  void init(Ts &&...ts) {
    derived().init(std::forward<Ts>(ts)...);
  }

  const Scalar TX() const { return TX_; }

  const Scalar omega(const Scalar &t) const {
    const Index idx = std::ceil(std::min(std::abs(t), TX_) / step_size_);
    const Index clamp = omegat_.size() - 1;
    return omegat_[std::min(idx, clamp)];
  }

  const Matrix &bb() const { return bb_; }
  const std::function<Scalar(const Vector &, const Vector &)> &dx() const {
    return dx_;
  }
  const std::function<Scalar(const Vector &, const Vector &)> &dy() const {
    return dy_;
  }

  const std::vector<Scalar> &tgrid() const { return tgrid_; }
  const std::vector<Scalar> &omegat() const { return omegat_; }

 protected:
  Matrix bb_;
  std::function<Scalar(const Vector &, const Vector &)> dx_;
  std::function<Scalar(const Vector &, const Vector &)> dy_;
  Scalar TX_;
  Scalar step_size_;
  std::vector<Scalar> tgrid_;
  std::vector<Scalar> omegat_;

  void setDistanceType(
      std::function<Scalar(const Vector &, const Vector &)> &df,
      std::string dist_type) {
    for (auto &chr : dist_type) chr = (char)toupper(chr);
    if (dist_type == "EUCLIDEAN") {
      df = [](const Vector &x, const Vector &y) { return (x - y).norm(); };
    } else
      assert(false && "desired distance not implemented");
    return;
  }
};

}  // namespace FMCA
#endif

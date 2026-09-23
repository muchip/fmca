// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2026, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_KERNEL_PDKERNEL_H_
#define FMCA_KERNEL_PDKERNEL_H_

namespace FMCA {

/**
 *  \brief positive definite kernels, string dispatched.
 *         k(x, y) = psi(s(x, y)) with psi from RadialFunctions of
 *         cpd_order == 0 and s(x, y) = ||x - y||^2 (RadialAdapter).
 **/
class PDKernel final : public PDKernelBase<PDKernel> {
 public:
  PDKernel() : eval_(nullptr), l_(0.), c_(0.) {}
  explicit PDKernel(const std::string &ktype, Scalar l = 1., Scalar c = 1.)
      : eval_(nullptr), l_(l), c_(c) {
    init(ktype);
  }
  // implicit copy/move/assignment: all members are values

  template <typename Derived, typename otherDerived>
  Scalar operator()(const MatrixBase<Derived> &x,
                    const MatrixBase<otherDerived> &y) const {
    return eval_(x, y, l_, c_);
  }

  std::string kernelType() const { return ktype_; }
  const Scalar &l() const { return l_; }
  const Scalar &c() const { return c_; }
  Scalar &l() { return l_; }
  Scalar &c() { return c_; }

 private:
  typedef Reference<const Vector> ConstVecRef;
  typedef Scalar (*EvalFn)(const ConstVecRef &, const ConstVecRef &, Scalar,
                           Scalar);

  template <typename RF>
  static Scalar call(const ConstVecRef &x, const ConstVecRef &y, Scalar l,
                     Scalar c) {
    return RadialAdapter<RF>::evaluate(x, y, l, c);
  }

  template <typename RF>
  void set() {
    eval_ = &call<RF>;
    return;
  }

  void init(const std::string &ktype) {
    ktype_ = ktype;
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    if (ktype_ == "EXPONENTIAL" || ktype_ == "MATERN12")
      set<RadialFunctions::Matern12>();
    else if (ktype_ == "MATERN32")
      set<RadialFunctions::Matern32>();
    else if (ktype_ == "MATERN52")
      set<RadialFunctions::Matern52>();
    else if (ktype_ == "MATERN72")
      set<RadialFunctions::Matern72>();
    else if (ktype_ == "MATERN92")
      set<RadialFunctions::Matern92>();
    else if (ktype_ == "GAUSSIAN" || ktype_ == "MATERNINF")
      set<RadialFunctions::MaternInf>();
    else if (ktype_ == "INVMULTIQUADRIC")
      set<RadialFunctions::InvMultiquadric>();
    else
      assert(false && "desired PD kernel not implemented");
    return;
  }

  EvalFn eval_;
  std::string ktype_;
  Scalar l_;
  Scalar c_;
};

using CovarianceKernel = PDKernel;

}  // namespace FMCA
#endif

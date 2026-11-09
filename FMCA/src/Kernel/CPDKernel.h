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
#ifndef FMCA_KERNEL_CPDKERNEL_H_
#define FMCA_KERNEL_CPDKERNEL_H_

namespace FMCA {

/**
 *  \brief conditionally positive definite kernels, string dispatched.
 *         k(x, y) = psi(s(x, y)) with psi from RadialFunctions of
 *         cpd_order >= 1 and s(x, y) = ||x - y||^2 (RadialAdapter).
 **/
class CPDKernel final : public CPDKernelBase<CPDKernel> {
 public:
  CPDKernel() : eval_(nullptr), cpd_order_(0), l_(0.), c_(0.) {}
  explicit CPDKernel(const std::string &ktype, Scalar l = 1., Scalar c = 1.)
      : eval_(nullptr), cpd_order_(0), l_(l), c_(c) {
    init(ktype);
  }
  // implicit copy/move/assignment: all members are values

  template <typename Derived, typename otherDerived>
  Scalar operator()(const MatrixBase<Derived> &x,
                    const MatrixBase<otherDerived> &y) const {
    return eval_(x, y, l_, c_);
  }

  Index cpd_order() const { return cpd_order_; }
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
    cpd_order_ = RF::cpd_order;
    return;
  }

  void init(const std::string &ktype) {
    ktype_ = ktype;
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    if (ktype_ == "MULTIQUADRIC")
      set<RadialFunctions::Multiquadric>();
    else if (ktype_ == "TPS1D")
      set<RadialFunctions::TPS1D>();
    else if (ktype_ == "TPS2D")
      set<RadialFunctions::TPS2D>();
    else if (ktype_ == "TPS3D")
      set<RadialFunctions::TPS3D>();
    else
      assert(false && "desired CPD kernel not implemented");
    return;
  }

  EvalFn eval_;
  Index cpd_order_;
  std::string ktype_;
  Scalar l_;
  Scalar c_;
};

}  // namespace FMCA
#endif

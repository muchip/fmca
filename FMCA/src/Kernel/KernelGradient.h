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
#ifndef FMCA_KERNEL_KERNELGRADIENT_H_
#define FMCA_KERNEL_KERNELGRADIENT_H_

namespace FMCA {

/**
 *  \brief first derivative of a radial kernel with respect to one
 *         component of the first argument, string dispatched:
 *         k(x, y) = d/dx_d psi(||x - y||^2). The string names the
 *         input kernel
 **/
class KernelGradient final : public KernelBase<KernelGradient> {
 public:
  KernelGradient() : eval_(nullptr), d_(0), l_(0.), c_(0.) {}
  KernelGradient(const std::string &ktype, Index d, Scalar l = 1.,
                 Scalar c = 1.)
      : eval_(nullptr), d_(d), l_(l), c_(c) {
    init(ktype);
  }
  // implicit copy/move/assignment: all members are values

  template <typename Derived, typename otherDerived>
  Scalar operator()(const MatrixBase<Derived> &x,
                    const MatrixBase<otherDerived> &y) const {
    assert(d_ < x.rows() && d_ < y.rows() &&
           "KernelGradient: component index out of range");
    return eval_(x, y, l_, c_);
  }

  std::string kernelType() const { return ktype_; }
  Index component() const { return d_; }
  const Scalar &l() const { return l_; }
  const Scalar &c() const { return c_; }
  Scalar &l() { return l_; }
  Scalar &c() { return c_; }

 private:
  typedef Reference<const Vector> ConstVecRef;
  typedef Scalar (*EvalFn)(const ConstVecRef &, const ConstVecRef &, Scalar,
                           Scalar);

  template <typename RF, int D>
  static Scalar call(const ConstVecRef &x, const ConstVecRef &y, Scalar l,
                     Scalar c) {
    return RadialAdapter<RadialFunctions::GradientOf<RF, D>>::evaluate(x, y, l,
                                                                       c);
  }

  template <typename RF, int D>
  void set() {
    eval_ = &call<RF, D>;
    return;
  }

  template <typename RF>
  void setComponent() {
    switch (d_) {
      case 0:
        set<RF, 0>();
        break;
      case 1:
        set<RF, 1>();
        break;
      case 2:
        set<RF, 2>();
        break;
      default:
        assert(false && "KernelGradient: component not instantiated");
    }
    return;
  }

  void init(const std::string &ktype) {
    ktype_ = ktype;
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    if (ktype_ == "MATERN32")
      setComponent<RadialFunctions::Matern32>();
    else if (ktype_ == "MATERN52")
      setComponent<RadialFunctions::Matern52>();
    else if (ktype_ == "MATERN72")
      setComponent<RadialFunctions::Matern72>();
    else if (ktype_ == "MATERN92")
      setComponent<RadialFunctions::Matern92>();
    else if (ktype_ == "GAUSSIAN" || ktype_ == "MATERNINF")
      setComponent<RadialFunctions::MaternInf>();
    else if (ktype_ == "INVMULTIQUADRIC")
      setComponent<RadialFunctions::InvMultiquadric>();
    else if (ktype_ == "MULTIQUADRIC")
      setComponent<RadialFunctions::Multiquadric>();
    else if (ktype_ == "TPS1D")
      setComponent<RadialFunctions::TPS1D>();
    else
      assert(false &&
             "gradient kernel not implemented: requires bounded psi' "
             "(Matern32+, Gaussian, (Inv)Multiquadric, TPS1D)");
    return;
  }

  EvalFn eval_;
  Index d_;
  std::string ktype_;
  Scalar l_;
  Scalar c_;
};
}  // namespace FMCA
#endif

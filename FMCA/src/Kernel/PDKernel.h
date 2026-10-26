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
 *         Alternatively, it offers access to tensor product kernels
 *         via TensorProductRadialAdapter.
 **/
template <typename MetricT = Metric::Euclidean>
class PDKernel final : public PDKernelBase<PDKernel<MetricT>> {
 public:
  typedef MetricT MetricType;
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

  template <typename Adapter>
  static Scalar call(const ConstVecRef &x, const ConstVecRef &y, Scalar l,
                     Scalar c) {
    return Adapter::evaluate(x, y, l, c);
  }
  template <typename Adapter>
  void set() {
    eval_ = &call<Adapter>;
  }

  void init(const std::string &ktype) {
    ktype_ = ktype;
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    if (ktype_ == "EXPONENTIAL" || ktype_ == "MATERN12")
      set<RadialAdapter<RadialFunctions::Matern12, MetricT>>();
    else if constexpr (std::is_same_v<MetricT, Metric::Euclidean>) {
      if (ktype_ == "MATERN32")
        set<RadialAdapter<RadialFunctions::Matern32>>();
      else if (ktype_ == "MATERN52")
        set<RadialAdapter<RadialFunctions::Matern52>>();
      else if (ktype_ == "MATERN72")
        set<RadialAdapter<RadialFunctions::Matern72>>();
      else if (ktype_ == "MATERN92")
        set<RadialAdapter<RadialFunctions::Matern92>>();
      else if (ktype_ == "GAUSSIAN" || ktype_ == "MATERNINF")
        set<RadialAdapter<RadialFunctions::MaternInf>>();
      else if (ktype_ == "INVMULTIQUADRIC")
        set<RadialAdapter<RadialFunctions::InvMultiquadric>>();
      else if (ktype_ == "TPEXPONENTIAL" || ktype_ == "TPMATERN12")
        set<TensorProductRadialAdapter<RadialFunctions::Matern12>>();
      else if (ktype_ == "TPMATERN32")
        set<TensorProductRadialAdapter<RadialFunctions::Matern32>>();
      else if (ktype_ == "TPMATERN52")
        set<TensorProductRadialAdapter<RadialFunctions::Matern52>>();
      else if (ktype_ == "TPMATERN72")
        set<TensorProductRadialAdapter<RadialFunctions::Matern72>>();
      else if (ktype_ == "TPMATERN92")
        set<TensorProductRadialAdapter<RadialFunctions::Matern92>>();
      else if (ktype_ == "TPINVMULTIQUADRIC")
        set<TensorProductRadialAdapter<RadialFunctions::InvMultiquadric>>();
      else
        assert(false && "desired PD kernel not implemented");
    } else
      assert(false && "desired PD kernel not implemented");
    return;
  }

  EvalFn eval_;
  std::string ktype_;
  Scalar l_;
  Scalar c_;
};

using CovarianceKernel = PDKernel<>;

}  // namespace FMCA
#endif

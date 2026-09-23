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
#ifndef FMCA_KERNEL_KERNELLAPLACIAN_H_
#define FMCA_KERNEL_KERNELLAPLACIAN_H_

namespace FMCA {
/**
 *  \brief negative Laplacian of a radial kernel, string dispatched:
 *         k(x, y) = -Laplace_x psi(||x - y||^2), realized as
 *         RadialAdapter<LaplacianOf<RF, DIM>>
 **/
class KernelLaplacian final : public PDKernelBase<KernelLaplacian> {
 public:
  KernelLaplacian() : eval_(nullptr), dim_(0), l_(0.), c_(0.) {}
  KernelLaplacian(const std::string &ktype, Index dim, Scalar l = 1.,
                  Scalar c = 1.)
      : eval_(nullptr), dim_(dim), l_(l), c_(c) {
    init(ktype);
  }
  // implicit copy/move/assignment: all members are values

  template <typename Derived, typename otherDerived>
  Scalar operator()(const MatrixBase<Derived> &x,
                    const MatrixBase<otherDerived> &y) const {
    assert(x.rows() == dim_ && y.rows() == dim_ &&
           "KernelLaplacian: dimension mismatch");
    return eval_(x, y, l_, c_);
  }

  std::string kernelType() const { return ktype_; }
  Index dim() const { return dim_; }
  const Scalar &l() const { return l_; }
  const Scalar &c() const { return c_; }
  Scalar &l() { return l_; }
  Scalar &c() { return c_; }

 private:
  typedef Reference<const Vector> ConstVecRef;
  typedef Scalar (*EvalFn)(const ConstVecRef &, const ConstVecRef &, Scalar,
                           Scalar);

  template <typename RF, int DIM>
  static Scalar call(const ConstVecRef &x, const ConstVecRef &y, Scalar l,
                     Scalar c) {
    return RadialAdapter<RadialFunctions::LaplacianOf<RF, DIM>>::evaluate(x, y,
                                                                          l, c);
  }

  template <typename RF, int DIM>
  void set() {
    eval_ = &call<RF, DIM>;
    return;
  }

  template <typename RF>
  void setDim() {
    switch (dim_) {
      case 1:
        set<RF, 1>();
        break;
      case 2:
        set<RF, 2>();
        break;
      case 3:
        set<RF, 3>();
        break;
      default:
        assert(false && "KernelLaplacian: dimension not instantiated");
        return;
    }
  }

  void init(const std::string &ktype) {
    ktype_ = ktype;
    for (auto &chr : ktype_) chr = (char)toupper(chr);
    if (ktype_ == "MATERN52")
      setDim<RadialFunctions::Matern52>();
    else if (ktype_ == "MATERN72")
      setDim<RadialFunctions::Matern72>();
    else if (ktype_ == "MATERN92")
      setDim<RadialFunctions::Matern92>();
    else if (ktype_ == "GAUSSIAN" || ktype_ == "MATERNINF")
      setDim<RadialFunctions::MaternInf>();
    else if (ktype_ == "INVMULTIQUADRIC")
      setDim<RadialFunctions::InvMultiquadric>();
    else if (ktype_ == "MULTIQUADRIC")
      setDim<RadialFunctions::Multiquadric>();
    else
      assert(false && "Kernel not implemented: requires bounded psi''");
    return;
  }

  EvalFn eval_;
  Index dim_;
  std::string ktype_;
  Scalar l_;
  Scalar c_;
};

}  // namespace FMCA
#endif

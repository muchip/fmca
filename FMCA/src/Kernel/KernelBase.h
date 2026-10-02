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
#ifndef FMCA_KERNEL_KERNELBASE_H_
#define FMCA_KERNEL_KERNELBASE_H_

namespace FMCA {
template <typename Derived>
struct KernelBase {
  KernelBase() {};
  //////////////////////////////////////////////////////////////////////////////
  // return a reference to the derived object
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

  template <typename otherDerived, typename other2Derived>
  Scalar operator()(const MatrixBase<otherDerived> &x,
                    const MatrixBase<other2Derived> &y) const {
    return derived().operator()(x, y);
  }

  Matrix eval(const Matrix &PR, const Matrix &PC) const {
    Matrix retval(PR.cols(), PC.cols());
    for (Index j = 0; j < PC.cols(); ++j)
      for (Index i = 0; i < PR.cols(); ++i)
        retval(i, j) = operator()(PR.col(i), PC.col(j));
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  std::string kernelType() const { return derived().kernelType(); }
};
}  // namespace FMCA
#endif

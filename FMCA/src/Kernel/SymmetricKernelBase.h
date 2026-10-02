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
#ifndef FMCA_KERNEL_SYMMETRICKERNELBASE_H_
#define FMCA_KERNEL_SYMMETRICKERNELBASE_H_

namespace FMCA {
template <typename Derived>
struct SymmetricKernelBase : public KernelBase<Derived> {
  typedef KernelBase<Derived> Base;
  using Base::operator();
  using Base::eval;
  using Base::kernelType;
  SymmetricKernelBase() {};
};
}  // namespace FMCA
#endif

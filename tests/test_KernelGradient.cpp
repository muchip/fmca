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

#include "../FMCA/Kernel"

template <typename RefKernel>
bool checkGradient(const std::string &ktype, FMCA::Index dim,
                   FMCA::Scalar l = 0.7, FMCA::Scalar c = 1.3) {
  const FMCA::Index npts = 20;
  const FMCA::Scalar h = 1e-5;
  const FMCA::Matrix X = FMCA::Matrix::Random(dim, npts);
  const FMCA::Matrix Y = FMCA::Matrix::Random(dim, npts);
  const RefKernel k(ktype, l, c);
  bool ok = true;
  for (FMCA::Index d = 0; d < dim; ++d) {
    const FMCA::KernelGradient kd(ktype, d, l, c);
    for (FMCA::Index i = 0; i < npts; ++i) {
      // first-order central difference of the full kernel:
      // d/dx_d k(x, y) ~ (k(x + h e_d, y) - k(x - h e_d, y)) / (2h)
      FMCA::Vector xp = X.col(i);
      FMCA::Vector xm = X.col(i);
      xp(d) += h;
      xm(d) -= h;
      const FMCA::Scalar fd = (k(xp, Y.col(i)) - k(xm, Y.col(i))) / (2. * h);
      const FMCA::Scalar val = kd(X.col(i), Y.col(i));
      const FMCA::Scalar err =
          std::abs(fd - val) / std::max(std::abs(val), FMCA::Scalar(1.));
      ok &= err < 1e-6;
    }
  }
  return ok;
}

int main() {
  bool ok = true;
  for (FMCA::Index dim : {1u, 2u, 3u}) {
    ok &= checkGradient<FMCA::PDKernel>("MATERN32", dim);
    ok &= checkGradient<FMCA::PDKernel>("MATERN52", dim);
    ok &= checkGradient<FMCA::PDKernel>("MATERN72", dim);
    ok &= checkGradient<FMCA::PDKernel>("MATERN92", dim);
    ok &= checkGradient<FMCA::PDKernel>("GAUSSIAN", dim);
    ok &= checkGradient<FMCA::PDKernel>("INVMULTIQUADRIC", dim);
    ok &= checkGradient<FMCA::CPDKernel>("MULTIQUADRIC", dim);
    ok &= checkGradient<FMCA::CPDKernel>("TPS1D", dim);
  }
  if (ok) std::cout << "PASSED" << std::endl;
  return ok ? 0 : 1;
}

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
bool checkLaplacian(const std::string &ktype, FMCA::Index dim,
                    FMCA::Scalar l = 0.7, FMCA::Scalar c = 1.3) {
  const FMCA::Index npts = 20;
  const FMCA::Scalar h = 1e-4;
  const FMCA::Matrix X = FMCA::Matrix::Random(dim, npts);
  const FMCA::Matrix Y = FMCA::Matrix::Random(dim, npts);
  const RefKernel k(ktype, l, c);
  const FMCA::KernelLaplacian kL(ktype, dim, l, c);
  bool ok = true;
  for (FMCA::Index i = 0; i < npts; ++i) {
    // second-order central difference of the full kernel, summed over
    // components: Delta_x k(x, y) ~ sum_d (k(x+h e_d) - 2k(x) + k(x-h e_d))/h^2
    FMCA::Scalar fd = 0.;
    const FMCA::Scalar k0 = k(X.col(i), Y.col(i));
    for (FMCA::Index d = 0; d < dim; ++d) {
      FMCA::Vector xp = X.col(i);
      FMCA::Vector xm = X.col(i);
      xp(d) += h;
      xm(d) -= h;
      fd += (k(xp, Y.col(i)) - 2. * k0 + k(xm, Y.col(i))) / (h * h);
    }
    // KernelLaplacian realizes the NEGATIVE Laplacian
    const FMCA::Scalar val = kL(X.col(i), Y.col(i));
    const FMCA::Scalar err =
        std::abs(fd + val) / std::max(std::abs(val), FMCA::Scalar(1.));
    ok &= err < 1e-5;
  }
  return ok;
}

int main() {
  bool ok = true;
  for (FMCA::Index dim : {1u, 2u, 3u}) {
    ok &= checkLaplacian<FMCA::PDKernel>("MATERN52", dim);
    ok &= checkLaplacian<FMCA::PDKernel>("MATERN72", dim);
    ok &= checkLaplacian<FMCA::PDKernel>("MATERN92", dim);
    ok &= checkLaplacian<FMCA::PDKernel>("GAUSSIAN", dim);
    ok &= checkLaplacian<FMCA::PDKernel>("INVMULTIQUADRIC", dim);
    ok &= checkLaplacian<FMCA::CPDKernel>("MULTIQUADRIC", dim);
  }
  if (ok) std::cout << "PASSED" << std::endl;
  return ok ? 0 : 1;
}

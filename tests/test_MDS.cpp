// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#include <Eigen/Dense>
#include <iostream>
#include <random>

#include "../FMCA/src/util/MDS.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  const FMCA::Index p = 100;
  const FMCA::Index d = 4;
  const FMCA::Index n = 1000;
  const FMCA::Scalar noise_lvl = 1e-10;

  const FMCA::Matrix P = FMCA::Matrix::Random(d, n);
  FMCA::Matrix Q(p, p);
  Q.setRandom();

  Eigen::HouseholderQR<FMCA::Matrix> qr(p, p);
  qr.compute(Q);
  Q = qr.householderQ();
  FMCA::Matrix Pp(p, n);
  Pp.setZero();
  Pp.topRows(d) = P;
  Pp = Q * Pp;
  Pp = Pp + noise_lvl * (2 * FMCA::Matrix::Random(p, n).array() - 1).matrix();
  FMCA::Matrix Dp(n, n);
  FMCA::Matrix Dd(n, n);
  for (FMCA::Index j = 0; j < n; ++j)
    for (FMCA::Index i = 0; i < n; ++i) {
      Dp(i, j) = (Pp.col(i) - Pp.col(j)).norm();
      Dd(i, j) = (P.col(i) - P.col(j)).norm();
    }
  FMCA::Matrix Pemb = FMCA::MDS(Dp, d);
  std::cout << "original dist: " << (Dp - Dd).norm() / Dd.norm() << std::endl;
  FMCA::Matrix Demb(n, n);
  for (FMCA::Index j = 0; j < n; ++j)
    for (FMCA::Index i = 0; i < n; ++i)
      Demb(i, j) = (Pemb.col(i) - Pemb.col(j)).norm();
  std::cout << "lost energy: " << (Demb - Dp).norm() / Dp.norm() << std::endl;
  std::cout << "embedding error: " << (Demb - Dd).norm() / Dd.norm()
            << std::endl;
  assert((Demb - Dd).norm() / Dd.norm() < 2 * noise_lvl && "error in MDS test");

  return 0;
}

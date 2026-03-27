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
//

#include "../FMCA/Clustering"
#include "../FMCA/ModulusOfContinuity"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"
#include <optional>

int main() {
  std::cout.setf(std::ios::unitbuf);

  FMCA::DiscreteModulusOfContinuity moc;
  FMCA::Tictoc T;
  FMCA::Index num_points = 1000;
  FMCA::Scalar moc_base_radius = 0.5;

  FMCA::Matrix P(4, num_points);
  FMCA::Matrix f(1, num_points);
  P.setRandom();
  P = 0.5 * (P.array() + 1);
  for (FMCA::Index i = 0; i < P.cols(); ++i)
    f(0, i) = std::sqrt(P(0, i));

  T.tic();
  moc.init(P, f, std::nullopt, moc_base_radius);
  T.toc("moc init: ");
  T.tic();
  FMCA::Matrix Omegat(moc.omegat().size(), 2);
  for (FMCA::Index i = 0; i < Omegat.rows(); ++i)
    Omegat.row(i) << moc.tgrid()[i], moc.omegat()[i];

  T.toc("moc queries: ");
  FMCA::IO::print2ascii("omegat.txt", Omegat);

  std::cout << "TX_: " << moc.TX();

  FMCA::EpsilonDiscreteModulusOfContinuity<FMCA::ClusterTree> emoc;
  T.tic();
  emoc.init(P, f, 1, moc_base_radius);
  T.toc("emoc init: ");
  T.tic();
  for (FMCA::Index i = 0; i < Omegat.rows(); ++i)
    Omegat.row(i) << moc.tgrid()[i], emoc.omega(moc.tgrid()[i], P, f);
  T.toc("emoc queries: ");
  FMCA::IO::print2ascii("eomegat.txt", Omegat);

  FMCA::FalconLSHDiscreteModulusOfContinuity lmoc;
  T.tic();
  lmoc.init(P, f, std::nullopt, moc_base_radius);
  T.toc("lshmoc init done: ");
  T.tic();
  for (FMCA::Index i = 0; i < Omegat.rows(); ++i)
    Omegat.row(i) << moc.tgrid()[i], lmoc.omega(moc.tgrid()[i], P, f);

  T.toc("lshmoc queries: ");
  FMCA::IO::print2ascii("lomegat.txt", Omegat);
  return 0;
}

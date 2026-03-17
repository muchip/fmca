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

int main() {
  FMCA::DiscreteModulusOfContinuity moc;
  FMCA::Tictoc T;
  FMCA::Index num_points = 1000;
  FMCA::Matrix P(1, num_points);
  FMCA::Matrix f(1, num_points);
  P.setRandom();
  P = 0.5 * (P.array() + 1);
  for (FMCA::Index i = 0; i < P.cols(); ++i)
    f(0, i) = std::sqrt(P(0, i));
  T.tic();
  moc.init(P, f, 1, 0.001);
  T.toc("moc init: ");

  FMCA::Matrix Omegat(moc.omegat().size(), 2);
  for (FMCA::Index i = 0; i < Omegat.rows(); ++i)
    Omegat.row(i) << moc.tgrid()[i], moc.omegat()[i];

  FMCA::IO::print2ascii("omegat.txt", Omegat);

  FMCA::EpsilonDiscreteModulusOfContinuity<FMCA::ClusterTree> emoc;
  T.tic();
  emoc.init(P, f, 1, 0.001);
  T.toc("emoc init: ");

  for (FMCA::Index i = 0; i < Omegat.rows(); ++i)
    Omegat.row(i) << moc.tgrid()[i], emoc.omega(moc.tgrid()[i], P, f);

  FMCA::IO::print2ascii("eomegat.txt", Omegat);

  FMCA::LSHDiscreteModulusOfContinuity lmoc;
  T.tic();
  lmoc.init(P, f, 1, 0.001);
  T.toc("lshmoc init: ");

  for (FMCA::Index i = 0; i < Omegat.rows(); ++i)
    Omegat.row(i) << moc.tgrid()[i], lmoc.omega(moc.tgrid()[i], P, f);

  FMCA::IO::print2ascii("lomegat.txt", Omegat);

  return 0;
}

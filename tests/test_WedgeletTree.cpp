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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Wedgelets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

#define DIM 3
#define NPTS 10000

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = FMCA::IO::ascii2Matrix("P.dat");
  const FMCA::Matrix rgb = FMCA::IO::ascii2Matrix("rgb.dat");
  std::cout << rgb.topRows(10) << std::endl << "......." << std::endl;
  FMCA::WedgeletTree<double> wt(P, 6);
  wt.computeWedges(P, rgb, 4, 1);
  FMCA::Vector hits(P.cols());
  hits.setZero();
  for (const auto &it : wt)
    if (!it.nSons() && it.block_size()) {
      for (FMCA::Index j = 0; j < it.block_size(); ++j) {
        assert(hits(it.indices()[j]) == 0 && "duplicate index");
        hits(it.indices()[j]) = 1;
      }
    }
  FMCA::Matrix retval = rgb;
  for (const auto &it : wt) {
    if (!it.nSons() && it.block_size()) {
      FMCA::MultiIndexSet<FMCA::TotalDegree> idcs(P.rows(), it.node().deg_);
      FMCA::Matrix VT(idcs.index_set().size(), it.block_size());
      for (FMCA::Index i = 0; i < it.block_size(); ++i)
        VT.col(i) =
            FMCA::internal::evalPolynomials(idcs, P.col(it.indices()[i]));
      const FMCA::Matrix eval = VT.transpose() * it.node().C_;
      for (FMCA::Index i = 0; i < it.block_size(); ++i)
        retval.row(it.indices()[i]) = eval.row(i);
    }
  }

  FMCA::Matrix lms = wt.landmarks(P);
  FMCA::Matrix lm3(3, lms.cols());
  lm3.setZero();
  lm3.topRows(2) = lms;
  FMCA::IO::plotPoints("lms.vtk", lm3);
  FMCA::Matrix P3(3, P.cols());
  P3.setZero();
  P3.topRows(2) = P;
  FMCA::IO::plotPointsColor("im.vtk", P3, rgb.col(0));
  FMCA::IO::plotPointsColor("cmp.vtk", P3, retval.col(0));
  FMCA::IO::plotPointsColor("cmp.vtk", P3,
                            (retval.col(0) - rgb.col(0)).cwiseAbs());

  assert(hits.sum() == hits.size() && "missing index");
  return 0;
}

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
//
#include "../FMCA/src/util/ActiveSetManager.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  FMCA::Matrix A(1000000, 1000);
  A.setRandom();
  std::vector<FMCA::Index> aidcs;
  std::vector<FMCA::Index> aidcs2;
  aidcs.push_back(1);
  aidcs.push_back(10);
  aidcs.push_back(0);
  aidcs.push_back(27);
  aidcs.push_back(500);
  aidcs.push_back(36);
  aidcs.push_back(20);
  aidcs.push_back(19);
  aidcs.push_back(777);
  aidcs.push_back(67);

  FMCA::ActiveSetManager astm(A, aidcs);
  aidcs2 = aidcs;
  aidcs2.push_back(666);
  aidcs2.push_back(555);
  aidcs2.push_back(333);
  aidcs2.push_back(222);
  astm.update(A, aidcs2);
  FMCA::Matrix Aactive(A.rows(), aidcs2.size());
  for (FMCA::Index i = 0; i < aidcs2.size(); ++i)
    Aactive.col(i) = A.col(aidcs2[i]);
  std::cout << "SVD error: "
            << (Aactive - astm.matrixU() * astm.sigma().asDiagonal() *
                              astm.matrixV().transpose())
                       .norm() /
                   Aactive.norm()
            << std::endl;
  FMCA::Matrix Aselect(A.rows(), aidcs.size());
  for (FMCA::Index i = 0; i < aidcs.size(); ++i)
    Aselect.col(i) = A.col(aidcs[i]);
  FMCA::Matrix ATA = Aselect.transpose() * Aselect;
  FMCA::Matrix V = astm.activeV(aidcs);
  FMCA::Matrix S = astm.sigma().asDiagonal();
  std::cout << "active err: "
            << (ATA - V * S * S * V.transpose()).norm() / ATA.norm()
            << std::endl;
  FMCA::Matrix VinvS = astm.activeVSinv(aidcs);
  std::cout << "inverse active err: "
            << (ATA * VinvS * VinvS.transpose() -
                FMCA::Matrix::Identity(ATA.rows(), ATA.cols()))
                       .norm() /
                   std::sqrt(ATA.rows())
            << std::endl;
  return 0;
}

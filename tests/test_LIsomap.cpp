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
extern "C" {
#include <metis.h>
}

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <random>
//
#include "../FMCA/Clustering"
#include "../FMCA/src/util/Graph.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/LIsomap.h"
#include "../FMCA/src/util/Tictoc.h"

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index npts = 1000;
  const FMCA::Index p = 100;
  const FMCA::Index d = 3;
  const FMCA::Scalar noise_lvl = 1e-3;

  FMCA::Matrix P = FMCA::Matrix::Random(d, npts);
  FMCA::Matrix Pp(p, npts);
  FMCA::Matrix Q(p, p);
  Q.setRandom();
  {
    Eigen::HouseholderQR<FMCA::Matrix> qr(p, p);
    qr.compute(Q);
    Q = qr.householderQ();
    Pp.setZero();
    Pp.topRows(d) = P;
    Pp = Q * Pp;
    Pp = Pp + noise_lvl * FMCA::Matrix::Random(p, npts);
  }
  FMCA::ClusterTree CT(Pp, 10);
  for (FMCA::Index kNN : {10, 50, 100, 500, 999}) {
    for (FMCA::Index M : {10, 50, 100, 500, 1000}) {
      std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, Pp, kNN);
      FMCA::Graph<idx_t, FMCA::Scalar> G;
      G.init(npts, A);
      FMCA::Scalar nrg = 0;
      FMCA::Matrix Pred = LIsomap(G, M, d, &nrg);
      FMCA::Scalar Fnorm2 = 0;
      FMCA::Scalar diff2_sum = 0;
      for (FMCA::Index i = 0; i < npts; ++i)
        for (FMCA::Index j = 0; j < i; ++j) {
          const FMCA::Scalar dist = (Pp.col(i) - Pp.col(j)).norm();
          const FMCA::Scalar emb_dist = (Pred.col(i) - Pred.col(j)).norm();
          Fnorm2 += dist * dist;
          diff2_sum += 2 * (emb_dist - dist) * (emb_dist - dist);
        }
      std::cout << std::left << std::setw(22)
                << "(k,M) = (" + std::to_string(kNN) + "," + std::to_string(M) +
                       ")"
                << "embedding error: " << std::sqrt(diff2_sum / Fnorm2)
                << std::endl;
    }
  }
  return 0;
}

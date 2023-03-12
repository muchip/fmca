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
#include <fstream>
#include <iostream>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/Tictoc.h"

template <typename Dists>
struct my_less {
  my_less(const Dists &dist) : dist_(dist) {}
  template <typename idx>
  bool operator()(const idx &a, const idx &b) const {
    return dist_(a) < dist_(b);
  }
  const Dists &dist_;
};

#define DIM 3
#define KMINS 10

int main(int argc, char *argv[]) {
  FMCA::Tictoc T;
  const FMCA::Index leaf_size = 10;
  for (FMCA::Index tests = 0; tests < 4; ++tests) {
    FMCA::Scalar told = 1;
    FMCA::Scalar tnew = 2;
    FMCA::Scalar avgfac = 0;
    for (FMCA::Index lvl = 4; lvl < 20; ++lvl) {
      const FMCA::Index npts = 1 << lvl;
      std::cout << npts << "\t " << lvl << "\t ";
      FMCA::Matrix P = FMCA::Matrix::Random(DIM, npts);
      FMCA::ClusterTree CT(P, leaf_size);
      T.tic();
      FMCA::iMatrix kMins = kNN(CT, P, KMINS);
      tnew = T.toc();
      std::cout << tnew << "sec.\t factor:" << tnew / told << std::endl;
      if (lvl > 4) avgfac += tnew / told;
      told = tnew;
#pragma omp parallel for
      for (auto j = 0; j < 100; ++j) {
        FMCA::Index the_index = rand() % P.cols();
        FMCA::Vector dist(P.cols());
        std::vector<FMCA::Index> idx(P.cols());
        std::iota(idx.begin(), idx.end(), 0);
        for (auto i = 0; i < P.cols(); ++i)
          dist(i) = (P.col(i) - P.col(the_index)).norm();
        std::sort(idx.begin(), idx.end(), my_less<FMCA::Vector>(dist));
        for (auto i = 1; i <= KMINS; ++i) {
          const FMCA::Scalar the_dist =
              (P.col(the_index) - P.col(kMins(the_index, i - 1))).norm();
          // assert(idx[i] == kMins(j, i - 1) && "index mismatch");
          assert(std::abs(the_dist - dist(idx[i])) < FMCA_ZERO_TOLERANCE &&
                 "dist mismatch");
        }
      }
    }
    std::cout << "average factor in this run: " << avgfac / 15 << std::endl;
  }
  return 0;
}

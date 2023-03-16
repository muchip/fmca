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

#include "../FMCA/Clustering"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/RandomTreeAccessor.h"

#define DIM 10
#define NPTS 10000000

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = Eigen::MatrixXd::Random(DIM, NPTS);
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.tic();
  const FMCA::ClusterTree CT(P, 10);
  T.toc("Tree setup: ");
  T.tic();
  const FMCA::internal::RandomTreeAccessor<FMCA::ClusterTree> rta(CT);
  T.toc("Accessor setup: ");
  FMCA::Index max_level = 0;
  for (auto &&node : CT) {
    max_level = node.level() < max_level ? max_level : node.level();
  }
  std::cout << max_level << " " << rta.max_level() << std::endl;
  std::cout << rta.nodes().size() << std::endl;
  for (auto &&it : rta.levels()) std::cout << it << std::endl;
  return 0;
}

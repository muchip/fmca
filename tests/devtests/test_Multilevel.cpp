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
#include <iostream>
//
#include <FMCA/Clustering>
//
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/NormalDistribution.h>
#include <FMCA/src/util/RandomTreeAccessor.h>
#include <FMCA/src/util/Tictoc.h>

//
#include <Eigen/Dense>

int main(int argc, char *argv[]) {
  const FMCA::Index npts = 10000000;
  FMCA::Tictoc T;
  FMCA::Matrix P(3, npts);
  FMCA::Vector signal(npts);
  FMCA::NormalDistribution nd(0, 1, 0);
  P.setRandom();
  for (FMCA::Index i = 0; i < npts; ++i) {
    P.col(i) = nd.randN(3, 1);
    P.col(i).normalize();
    signal(i) = (std::abs(P(0, i)));
  }
  // FMCA::IO::plotPointsColor("signal.vtk", P, signal);
  FMCA::ClusterTree ct(P, 10);
  FMCA::internal::RandomTreeAccessor<FMCA::ClusterTree> rta;
  T.tic();
  rta.init(ct);
  T.toc("tree mapped: ");
  std::vector<FMCA::Index> s_pattern(rta.nnodes());
  std::vector<FMCA::Scalar> s_values(rta.nnodes());
  for (auto it = rta.nodes().rbegin(); it != rta.nodes().rend(); ++it) {
    const FMCA::ClusterTree &node = *(*it);
    FMCA::Index sample_index = 0;
    if (node.nSons()) {
      sample_index = node.indices()[rand() % node.block_size()];
      // s_pattern[node.sons(0).block_id()];
    } else {
      if (node.block_size())
        sample_index = node.indices()[rand() % node.block_size()];
    }
    s_pattern[node.block_id()] = sample_index;
  }
  FMCA::Scalar max_value = 0;
  for (FMCA::Index i = rta.levels()[0]; i < rta.levels()[1]; ++i) {
    s_values[i] = signal[s_pattern[rta.nodes()[i]->block_id()]];
    max_value =
        max_value < std::abs(s_values[i]) ? std::abs(s_values[i]) : max_value;
  }
  std::cout << max_value << std::endl;
  std::cout << std::endl;
  for (FMCA::Index l = 1; l <= rta.max_level(); ++l) {
    max_value = 0;
    for (FMCA::Index i = rta.levels()[l]; i < rta.levels()[l + 1]; ++i) {
      if (rta.nodes()[i]->block_size()) {
        const FMCA::Index dad_id = (rta.nodes()[i]->dad()).block_id();
        s_values[i] = signal[s_pattern[rta.nodes()[i]->block_id()]] -
                      signal[s_pattern[dad_id]];
        max_value = max_value < std::abs(s_values[i]) ? std::abs(s_values[i])
                                                      : max_value;
      }
    }
    std::cout << max_value << std::endl;
  }

  return 0;
}

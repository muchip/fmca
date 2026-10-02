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
#include <FMCA/src/util/IO.h>
#include <FMCA/src/util/Tictoc.h>

#include <Eigen/Dense>
#include <FMCA/Clustering>
#include <iostream>

#define DIM 3
#define NPTS 1000000

int main() {
  FMCA::Tictoc T;

  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = FMCA_INF;
  const FMCA::Matrix P = FMCA::Matrix::Random(DIM, NPTS);
  std::cout
      << "Cluster splitter:             "
      << FMCA::internal::traits<FMCA::ClusterTree>::Splitter::splitterName()
      << std::endl;
  T.tic();
  for (auto i = 0; i < 10; ++i) {
    FMCA::iVector index_hits(P.cols());
    index_hits.setZero();
    FMCA::Index leaf_size = rand() % 200 + 5;
    FMCA::ClusterTree CT(P, leaf_size);
    for (i = 0; i < P.cols(); ++i) index_hits(CT.indices()[i]) = 1;
    assert(index_hits.sum() == P.cols() && "CT lost indices");
    std::vector<const FMCA::ClusterTree *> stack{&CT};
    while (stack.size()) {
      const auto &node = *stack.back();
      stack.pop_back();
      std::set<FMCA::Index> pidx(node.indices(),
                                 node.indices() + node.block_size());
      std::set<FMCA::Index> cidx;
      for (auto i = 0; i < node.nSons(); ++i) {
        cidx.insert(node.sons(i).indices(),
                    node.sons(i).indices() + node.sons(i).block_size());
        stack.push_back(&node.sons(i));
      }
      assert((!node.nSons() || pidx == cidx) &&
             "parent indices != union of sons");
    }
  }
  T.toc("construction of 10 cluster trees: ");
  return 0;
}

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
#include <Eigen/Dense>
#include <iostream>

#include "../FMCA/Clustering"
#include "../FMCA/src/util/FibonacciLattice.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  const FMCA::Matrix P = FMCA::FibonacciLattice(100000);
  std::cout << "Cluster splitter:             "
            << FMCA::internal::traits<
                   FMCA::SphereClusterTree>::Splitter::splitterName()
            << std::endl;
  T.tic();
  for (auto i = 0; i < 10; ++i) {
    FMCA::iVector index_hits(P.cols());
    index_hits.setZero();
    FMCA::Index leaf_size = rand() % 200 + 5;
    FMCA::SphereClusterTree CT(P, leaf_size);
    for (auto j = 0; j < P.cols(); ++j) index_hits(CT.indices()[j]) = 1;
    assert(index_hits.sum() == P.cols() && "CT lost indices");
    std::vector<const FMCA::SphereClusterTree *> stack{&CT};
    while (stack.size()) {
      const auto &node = *stack.back();
      stack.pop_back();
      // index sets: father is the union of its sons
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
      // all points of a cluster lie in its spherical cap
      for (auto j = 0; j < node.block_size(); ++j)
        assert(FMCA::Metric::GeodesicSphere::d(node.center(),
                                               P.col(node.indices()[j])) <=
                   node.radius() + 1e3 * FMCA_ZERO_TOLERANCE &&
               "point outside spherical cap");
    }
  }
  T.toc("construction of 10 sphere cluster trees: ");
  return 0;
}

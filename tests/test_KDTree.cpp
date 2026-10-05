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
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"

int main() {
  FMCA::Tictoc T;
  FMCA::Scalar fill_distance = 0;
  FMCA::Scalar separation_radius = 1. / 0.;
  FMCA::Matrix P(3, 8 * 8 * 8);
  FMCA::iVector iidcs = FMCA::iVector::LinSpaced(8, 0, 7);
  FMCA::Index l = 0;
  for (FMCA::Index i = 0; i < 8; ++i) {
    for (FMCA::Index j = 0; j < 8; ++j) {
      for (FMCA::Index k = 0; k < 8; ++k) {
        P.col(l) << iidcs(i), iidcs(j), iidcs(k);
        ++l;
      }
    }
  }
  P = P + 0.1 * Eigen::MatrixXd::Random(3, 8 * 8 * 8);
  std::cout << "Cluster splitter:             "
            << FMCA::internal::traits<FMCA::KDTree>::Splitter::splitterName()
            << std::endl;
  T.tic();
  for (auto i = 0; i < 10; ++i) {
    FMCA::iVector index_hits(P.cols());
    index_hits.setZero();
    FMCA::Index leaf_size = rand() % 200 + 5;
    FMCA::KDTree CT(P, leaf_size);
    for (auto j = 0; j < P.cols(); ++j) index_hits(CT.indices()[j]) = 1;
    assert(index_hits.sum() == P.cols() && "CT lost indices");
    std::vector<const FMCA::KDTree *> stack{&CT};
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

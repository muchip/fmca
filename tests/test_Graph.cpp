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

//
#include "../FMCA/Clustering"
#include "../FMCA/src/util/Graph.h"
#include "../FMCA/src/util/IO.h"
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
  const FMCA::Index npts = 1000;
  const FMCA::Index leaf_size = 10;

  std::mt19937 mt;
  mt.seed(0);
  FMCA::Matrix P(3, npts);
  {
    std::normal_distribution<FMCA::Scalar> dist(0.0, 1.0);
    for (FMCA::Index i = 0; i < P.cols(); ++i) {
      P.col(i) << dist(mt), dist(mt), dist(mt);
      P.col(i) /= P.col(i).norm();
    }
  }
  FMCA::ClusterTree CT(P, leaf_size);
  T.tic();
  std::vector<Eigen::Triplet<FMCA::Scalar>> A = FMCA::symKNN(CT, P, 10);
  T.toc("kNN:");

  FMCA::Graph<idx_t, FMCA::Scalar> G;
  G.init(npts, A);
  G.print("graph0.vtk", P);
  T.tic();
  FMCA::Matrix D = G.distanceMatrix();
  T.toc("Floyd Warshall: ");
  std::vector<FMCA::Index> idcs(npts);
  std::iota(idcs.begin(), idcs.end(), 0);
  T.tic();
  auto DDijkstra = G.partialDistanceMatrix(idcs);
  T.toc("Dijkstra: ");
  for (FMCA::Index j = 0; j < G.nnodes(); ++j)
    for (FMCA::Index i = 0; i < G.nnodes(); ++i) {
      assert(std::abs(D(i, j) - DDijkstra[i][j]) < 1e3 * FMCA_ZERO_TOLERANCE ||
             D(i, j) == DDijkstra[i][j]);
    }
  std::vector<idx_t> part = FMCA::METIS::partitionGraph(G);
  FMCA::Graph<idx_t, FMCA::Scalar> G1 = G.split(part);
  G1.print("graph1.vtk", P);
  G.print("graph2.vtk", P);
  FMCA::IO::plotPointsColor(
      "points.vtk", P, Eigen::Map<Eigen::VectorXi>(part.data(), part.size()));
  part = FMCA::METIS::partitionGraph(G);
  G1 = G.split(part);
  G1.print("graph3.vtk", P);
  G.print("graph4vtk", P);
  // check distance matrices

  FMCA::Graph<idx_t, FMCA::Scalar> G2;
  A.clear();
  G.init(5, A);
  std::cout << FMCA::Matrix(G.graph()) << std::endl << "---------\n";
  std::cout << G.distanceMatrix() << std::endl << "=========\n";
  A.clear();
  for (FMCA::Index i = 0; i < 5; ++i)
    for (FMCA::Index j = 0; j < i; ++j) {
      FMCA::Scalar w = 0.5 * (rand() % 3);
      A.push_back(Eigen::Triplet<FMCA::Scalar>(i, j, w));
      A.push_back(Eigen::Triplet<FMCA::Scalar>(j, i, w));
    }
  G.init(5, A);
  std::cout << FMCA::Matrix(G.graph()) << std::endl << "---------\n";
  std::cout << G.distanceMatrix() << std::endl;
  return 0;
}

#include <Eigen/Dense>
#include <fstream>

#include "FMCA/BlockClusterTree"
#include "FMCA/Samplets"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/MultiIndexSet.h"
#include "util/tictoc.hpp"

#define NPTS 1e2
#define DIM 3

using ClusterT = FMCA::ClusterTree<double, DIM, 10>;

int main() {
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  // P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  for (auto i = 0; i < P.cols(); ++i)
    P.col(i) *= 1 / nrms(i);
  tictoc T;
  T.tic();
  ClusterT CT(P);
  FMCA::SampletTree<ClusterT> ST(P, CT, 2);
  T.toc("set up ct: ");
  std::vector<std::vector<int>> tree;
  CT.exportTreeStructure(tree);
  for (auto i = 0; i < tree.size(); ++i) {
    int numInd = 0;
    for (auto j = 0; j < tree[i].size(); ++j)
      numInd += tree[i][j];
    std::cout << i << ") " << tree[i].size() << " " << numInd << "\n";
  }
  std::vector<ClusterT *> leafs;
  CT.getLeafIterator(leafs);
  int numInd = 0;
  for (auto i = 0; i < leafs.size(); ++i)
    numInd += (leafs[i])->get_indices().size();
  std::cout << leafs.size() << " " << numInd << "\n";
  for (auto level = 0; level < 14; ++level) {
    std::vector<Eigen::Matrix3d> bbvec;
    CT.get_BboxVector(&bbvec, level);
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<Eigen::Matrix3d> bbvec;
  CT.get_BboxVectorLeafs(&bbvec);
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPoints("points.vtk", P);

  return 0;
}

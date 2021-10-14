#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "FMCA/ClusterTree"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/tictoc.hpp"

//using ClusterT = FMCA::ClusterTree<double, FMCA::ClusterSplitter::GeometricBisection<double>>;
using ClusterT = FMCA::ClusterTree<double>;

int main() {
  std::cout << "using random points\n";
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(3, 100000);
  std::cout << P.rows() << " " << P.cols() << std::endl;
  P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  for (auto i = 0; i < P.cols(); ++i) P.col(i) *= 1 / nrms(i);
  tictoc T;

  T.tic();
  ClusterT CT(P, 100);
  T.toc("set up cluster tree: ");
  {
    std::vector<std::vector<FMCA::IndexType>> tree;
    CT.exportTreeStructure(tree);
    std::cout << "cluster structure: " << std::endl;
    std::cout << "l)\t#pts\ttotal#pts" << std::endl;
    for (auto i = 0; i < tree.size(); ++i) {
      int numInd = 0;
      for (auto j = 0; j < tree[i].size(); ++j) numInd += tree[i][j];
      std::cout << i << ")\t" << tree[i].size() << "\t" << numInd << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
  }

  {
    std::vector<const ClusterT *> leafs;
    CT.getLeafIterator(leafs);
    int numInd = 0;
    for (auto i = 0; i < leafs.size(); ++i)
      numInd += (leafs[i])->get_indices().size();
    for (auto level = 0; level < 14; ++level) {
      std::vector<Eigen::MatrixXd> bbvec;
      CT.get_BboxVector(&bbvec, level);
      FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
    }
    std::vector<Eigen::MatrixXd> bbvec;
    CT.get_BboxVectorLeafs(&bbvec);
    FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
    FMCA::IO::plotPoints("points.vtk", P);
  }
  return 0;
}

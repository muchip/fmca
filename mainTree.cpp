#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "FMCA/Clustering"
#include "FMCA/H2Matrix"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/tictoc.hpp"

int main() {
  std::cout << "using random points\n";
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(3, 1379);
  std::cout << P.rows() << " " << P.cols() << std::endl;
  P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  for (auto i = 0; i < P.cols(); ++i) P.col(i) *= 1 / nrms(i);
  tictoc T;

  T.tic();
  FMCA::ClusterT CT(P, 2);
  FMCA::H2ClusterTree H2T(P, 2);
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
  int oldl = 0;
  int numInd = 0;
  int i = 0;
  for (const auto &n : H2T) {
    if (oldl != n.level()) {
      std::cout << oldl << ")\t" << i << "\t" << numInd << std::endl;
      i = 0;
      numInd = 0;
      oldl = n.level();
    }
    numInd += n.node().indices_.size();
    ++i;
  }
  std::cout << oldl << ")\t" << i << "\t" << numInd << std::endl;
  {
    std::vector<const FMCA::TreeBase<FMCA::ClusterT> *> leafs;
    for (auto level = 0; level < 14; ++level) {
      std::vector<Eigen::MatrixXd> bbvec;
      for (const auto &node : CT) {
        if (node.level() == level) bbvec.push_back(node.derived().bb());
      }
      FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
    }
    std::vector<Eigen::MatrixXd> bbvec;

    for (const auto &node : CT) {
      if (!node.nSons()) bbvec.push_back(node.derived().bb());
    }
    FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
    FMCA::IO::plotPoints("points.vtk", P);
  }
  return 0;
}
#if 0
  //////////////////////////////////////////////////////////////////////////////
  // get a vector with the bounding boxes on a certain level
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVector(std::vector<eigenMatrix> *bbvec, IndexType lvl = 0) {
    if (nSons() && level() < lvl)
      for (auto i = 0; i < nSons(); ++i)
        sons(i).get_BboxVector(bbvec, lvl);
    if (level() == lvl)
      if (node().indices_.size())
        bbvec->push_back(node().bb_);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  void get_BboxVectorLeafs(std::vector<eigenMatrix> *bbvec) {
    if (nSons())
      for (auto i = 0; i < nSons(); ++i)
        sons(i).get_BboxVectorLeafs(bbvec);
    else if (node().indices_.size())
      bbvec->push_back(node().bb_);
    return;
  }
#endif

#define USE_QR_CONSTRUCTION_
#define FMCA_CLUSTERSET_
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/H2Matrix/TensorProductInterpolation.h"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"

struct exponentialKernel {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<Derived> &y) const {
    return exp(-10 * (x - y).norm());
  }
};
////////////////////////////////////////////////////////////////////////////////
//#define NPTS 5000
//#define NPTS 131072
//#define NPTS 65536
//#define NPTS 32768
//#define NPTS 16384
#define NPTS 100000
//#define NPTS 4096
//#define NPTS 2048
//#define NPTS 1024
//#define NPTS 512
//#define NPTS 64
//#define PLOT_BOXES_
//#define TEST_H2MATRIX_
//#define TEST_COMPRESSOR_
//#define TEST_SAMPLET_TRANSFORM_
//#define TEST_VANISHING_MOMENTS_
//#define USE_BUNNY_

int main() {
  //////////////////////////////////////////////////////////////////////////////
  tictoc T;
  Eigen::Matrix3d rot;
  rot << 0.8047379, -0.3106172, 0.5058793, 0.5058793, 0.8047379, -0.3106172,
      -0.3106172, 0.5058793, 0.8047379;
  Eigen::MatrixXd P(3, NPTS);
  for (auto i = 0; i < NPTS; ++i)
        P.col(i) << cos(4 * FMCA_PI * double(i) / NPTS),
           sin(4 * FMCA_PI * double(i) / NPTS), 8 * double(i) / NPTS;
    //P.col(i) << double(i) / NPTS, 0, 0;
  P = rot * P;
  T.tic();
  FMCA::ClusterT CT(P, 10);
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
  T.tic();
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
  T.toc();
  return 0;
}

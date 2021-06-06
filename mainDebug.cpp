#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/BlockClusterTree"
#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"

//#define NPTS 16384
//#define NPTS 8192
//#define NPTS 1800
#define NPTS 1000
//#define NPTS 512
#define DIM 2
#define TEST_SAMPLET_TRANSFORM_
#define TEST_COMPRESSOR_

struct Gaussian {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-10 * (x - y).norm());
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, 1>;

int main() {
#if 0
  std::cout << "loading data: ";
  Eigen::MatrixXd B = readMatrix("bunny.txt");
  std::cout << "data size: ";
  std::cout << B.rows() << " " << B.cols() << std::endl;
  std::cout << "----------------------------------------------------\n";
  Eigen::MatrixXd P = B.transpose();
#else
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
//  Eigen::VectorXd nrms = P.colwise().norm();
//  for (auto i = 0; i < P.cols(); ++i)
//    P.col(i) *= 1 / nrms(i);
#endif
  tictoc T;
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, 2);
  T.toc("set up samplet tree: ");
  std::cout << "----------------------------------------------------\n";
  //////////////////////////////////////////////////////////////////////////////
  std::vector<std::vector<FMCA::IndexType>> tree;
  CT.exportTreeStructure(tree);
  std::cout << "cluster structure: " << std::endl;
  std::cout << "l)\t#pts\ttotal#pts" << std::endl;
  for (auto i = 0; i < tree.size(); ++i) {
    int numInd = 0;
    for (auto j = 0; j < tree[i].size(); ++j)
      numInd += tree[i][j];
    std::cout << i << ")\t" << tree[i].size() << "\t" << numInd << "\n";
  }
  std::cout << "----------------------------------------------------\n";
  //////////////////////////////////////////////////////////////////////////////
#ifdef PLOT_BOXES_
  std::vector<ClusterT *> leafs;
  CT.getLeafIterator(leafs);
  int numInd = 0;
  for (auto i = 0; i < leafs.size(); ++i)
    numInd += (leafs[i])->get_indices().size();
  for (auto level = 0; level < 14; ++level) {
    std::vector<Eigen::Matrix3d> bbvec;
    CT.get_BboxVector(&bbvec, level);
    FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
  std::vector<Eigen::Matrix<double, DIM, 3u>> bbvec;
  CT.get_BboxVectorLeafs(&bbvec);
  FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  FMCA::IO::plotPoints("points.vtk", P);
#endif
#ifdef TEST_COMPRESSOR_
  {
    T.tic();
    FMCA::BivariateCompressor<FMCA::SampletTree<ClusterT>> BC(P, ST,
                                                              Gaussian());
    T.toc("set up compression pattern: ");

    std::cout << "----------------------------------------------------\n";
    Eigen::MatrixXd K(P.cols(), P.cols());
    auto fun = Gaussian();
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) = fun(P.col(CT.get_indices()[i]), P.col(CT.get_indices()[j]));
    Eigen::SparseMatrix<double> Tmat = ST.get_transformationMatrix();
    Eigen::MatrixXd SK = Tmat * K * Tmat.transpose();
    Bembel::IO::print2m("Smatrix.m", "S", BC.get_S(P.cols()), "w");
    Bembel::IO::print2m("S2matrix.m", "S2", SK, "w");
  }
#endif
//////////////////////////////////////////////////////////////////////////////
#ifdef TEST_SAMPLET_TRANSFORM_
  {
    eigen_assert(P.cols() <= 2e4 &&
                 "Test samplet transform only for small matrices");
    T.tic();
    Eigen::SparseMatrix<double> Tmat = ST.get_transformationMatrix();
    T.toc("time samplet transform test: ");
    std::cout << "error: "
              << (Tmat.transpose() * Tmat -
                  Eigen::MatrixXd::Identity(Tmat.rows(), Tmat.cols()))
                         .norm() /
                     sqrt(Tmat.rows())
              << std::endl;
    std::cout << "----------------------------------------------------\n";
    Eigen::MatrixXd K(P.cols(), P.cols());
    auto fun = Gaussian();
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) = fun(P.col(CT.get_indices()[i]), P.col(CT.get_indices()[j]));
    Eigen::MatrixXd K2 = K;
    Eigen::MatrixXd SK = Tmat * K * Tmat.transpose();
    T.tic();
    ST.sampletTransformMatrix(K);
    T.toc("time samplet transform matrix: ");
    std::cout << "error: " << (K - SK).norm() / SK.norm() << std::endl;
    T.tic();
    ST.inverseSampletTransformMatrix(K);
    T.toc("time inverse samplet transform matrix: ");
    std::cout << "error: " << (K - K2).norm() / K2.norm() << std::endl;
    std::cout << "----------------------------------------------------\n";
  }
#endif

  return 0;
}

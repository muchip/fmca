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
//#define NPTS 2048
//#define NPTS 1024
#define NPTS 512
#define DIM 3
#define TEST_SAMPLET_TRANSFORM_

struct Gaussian {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-20 * (x - y).norm());
  }
};

using ClusterT = FMCA::ClusterTree<double, DIM, 2>;

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
  FMCA::SampletTree<ClusterT> ST(CT, 1);
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
  T.tic();
  FMCA::BivariateCompressor<FMCA::SampletTree<ClusterT>> BC(ST, Gaussian());
  T.toc("set up compression pattern: ");
  T.tic();
  Eigen::MatrixXd S(P.cols(), P.cols());
  S.setZero();
  Eigen::MatrixXd C =
      BC.recursivelyComputeBlock(&S, ST, ST, Gaussian(), true, true);
  T.toc("wavelet transform: ");
  Bembel::IO::print2m("Smatrix.m", "S", S, "w");
  std::cout << "----------------------------------------------------\n";
//////////////////////////////////////////////////////////////////////////////
#ifdef TEST_SAMPLET_TRANSFORM_
  T.tic();
  Eigen::MatrixXd Tmat(P.cols(), P.cols());
  Eigen::VectorXd unit(P.cols());
  auto idcs = CT.get_indices();
  double inv_err = 0;
  for (auto i = 0; i < P.cols(); ++i) {
    unit.setZero();
    unit(idcs[i]) = 1;
    Tmat.col(i) = ST.sampletTransform(unit);
    Eigen::VectorXd bt_unit = ST.inverseSampletTransform(Tmat.col(i));
    inv_err += (bt_unit - unit).squaredNorm();
  }
  std::cout << "transform correct? error: "
            << (Tmat.transpose() * Tmat -
                Eigen::MatrixXd::Identity(Tmat.rows(), Tmat.cols()))
                       .norm() /
                   sqrt(Tmat.rows())
            << " " << sqrt(inv_err / Tmat.cols()) << std::endl;
  T.toc("time samplet transform test: ");
  std::cout << "----------------------------------------------------\n";
  Eigen::MatrixXd K(CT.get_indices().size(), CT.get_indices().size());
  auto fun = Gaussian();
  for (auto j = 0; j < CT.get_indices().size(); ++j)
    for (auto i = 0; i < CT.get_indices().size(); ++i)
      K(i, j) = fun(P.col(CT.get_indices()[i]), P.col(CT.get_indices()[j]));
  Eigen::MatrixXd SK = Tmat.transpose() * K * Tmat;
  std::cout << C.rows() << " " << C.cols() << std::endl;
  std::cout << SK.rows() << " " << SK.cols() << std::endl;
  Bembel::IO::print2m("S2matrix.m", "S2", SK, "w");
  // std::cout << (C - SK).norm() / SK.norm() << std::endl;
#endif
#if 0
    std::function<double(const Eigen::VectorXd &)> fun =
      [](const Eigen::VectorXd &x) { return exp(-10 * x.squaredNorm()); };
  auto fdata = FMCA::functionEvaluator<ClusterT>(P, CT, fun);
  Kmat.resize(P.cols(), P.cols());
  TWmat1.resize(P.cols(), P.cols());
  TWmat2.resize(P.cols(), P.cols());
  // generate transformation matrix
  for (auto i = 0; i < P.cols(); ++i) {
    unit.setZero();
    unit(idcs[i]) = 1;
    Tmat.col(i) = ST.sampletTransform(unit);
    for (auto j = 0; j < P.cols(); ++j)
      Kmat(j, i) = exp(-10 * (P.col(idcs[j]) - P.col(idcs[i])).norm());
    TWmat1.col(i) = ST.sampletTransform(Kmat.col(i));
  }

  std::cout << "transform correct? "
            << (Tmat.transpose() * Tmat -
                Eigen::MatrixXd::Identity(Tmat.rows(), Tmat.cols()))
                       .norm() /
                   sqrt(Tmat.rows())
            << std::endl;
  for (auto i = 0; i < P.cols(); ++i)
    TWmat2.col(i) = ST.sampletTransform(TWmat1.transpose().col(i));

  for (auto i = 0; i < P.cols(); ++i)
    for (auto j = 0; j < P.cols(); ++j)
      TWmat2(j, i) = abs(TWmat2(j, i)) > 1e-4 ? TWmat2(j, i) : 0;

  Eigen::SparseMatrix<double> TWs = TWmat2.sparseView();

  // Bembel::IO::print2m("Tmat.m", "T", Tmat, "w");
  // Bembel::IO::print2m("Kmat.m", "K", Kmat, "w");
  Bembel::IO::print2spascii("KWmat.txt", TWs, "w");
  Bembel::IO::print2m("comp.m", "fwt", ST.sampletTransform(fdata), "w");
  std::vector<Eigen::Matrix3d> bbvec;
  // CT.get_BboxVectorLeafs(&bbvec);
  // FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
  // FMCA::IO::plotPoints<ClusterT>("points.vtk", CT, P, fdata);
#endif
  return 0;
}

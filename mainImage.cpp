#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/BlockClusterTree"
#include "FMCA/Samplets"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "imgCompression/matrixReader.h"
#include "print2file.hpp"
#include "util/tictoc.hpp"
#define DIM 3

using ClusterT = FMCA::ClusterTree<double, DIM, 3>;

int main() {
  Eigen::MatrixXd Rchan = readMatrix("imgCompression/Rchan.txt");
  Eigen::MatrixXd Gchan = readMatrix("imgCompression/Gchan.txt");
  Eigen::MatrixXd Bchan = readMatrix("imgCompression/Bchan.txt");
  std::cout << "img data: " << Rchan.rows() << "x" << Rchan.cols() << std::endl;
  Eigen::MatrixXd P;
  P.resize(3, Rchan.rows() * Rchan.cols());
  auto k = 0;
  for (auto i = 0; i < Rchan.cols(); ++i)
    for (auto j = 0; j < Rchan.rows(); ++j) {
      P(0, k) = i;
      P(1, k) = j;
      P(2, k) = 0;
      ++k;
    }
  ClusterT CT(P);
  FMCA::SampletTree<ClusterT> ST(P, CT, 0);
  auto indices = CT.get_indices();
  Eigen::MatrixXd Tmat;
  Tmat.resize(P.cols(), P.cols());
  Eigen::VectorXd unit(P.cols());
  // generate transformation matrix
  for (auto i = 0; i < P.cols(); ++i) {
    unit.setZero();
    unit(indices[i]) = 1;
    Tmat.col(i) = ST.sampletTransform(unit);
  }
  Eigen::Map<Eigen::VectorXd> bla(Rchan.data(), Rchan.rows() * Rchan.cols());
  Eigen::VectorXd fdata(indices.size());
  for (auto i = 0; i < indices.size(); ++i) {
    fdata(i) = bla(indices[i]);
  }
  Eigen::VectorXd compf = Tmat * fdata;
  for (auto i = 0; i < compf.size(); ++i)
    compf(i) = abs(compf(i)) > 1e-4 ? compf(i) : 0;
  compf = Tmat.transpose() * compf;
  std::vector<FMCA::IndexType> rev_indices;
  std::cout << indices.size() << std::endl;
  std::cout << P.rows() << " " << P.cols() << std::endl;
  rev_indices.resize(indices.size());
  for (auto i = 0; i < indices.size(); ++i)
    rev_indices[indices[i]] = i;
  FMCA::IO::plotPoints<ClusterT>("points.vtk", CT, P, fdata);
  FMCA::IO::plotPoints<ClusterT>("pointsComp.vtk", CT, P, compf);
#if 0
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  // P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  // for (auto i = 0; i < P.cols(); ++i)
  //  P.col(i) *= 1 / nrms(i);
  tictoc T;
  T.tic();
  ClusterT CT(P);
  FMCA::SampletTree<ClusterT> ST(P, CT, 0);
  T.toc("set up ct: ");
  std::vector<std::vector<FMCA::IndexType>> tree;
  CT.exportTreeStructure(tree);
  for (auto i = 0; i < tree.size(); ++i) {
    int numInd = 0;
    for (auto j = 0; j < tree[i].size(); ++j)
      numInd += tree[i][j];
    std::cout << i << ") " << tree[i].size() << " " << numInd << "\n";
  }
#if 0
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
#endif
  std::function<double(const Eigen::VectorXd &)> fun =
      [](const Eigen::VectorXd &x) { return exp(-10 * x.squaredNorm()); };
  auto fdata = FMCA::functionEvaluator<ClusterT>(P, CT, fun);

  Eigen::MatrixXd Tmat, Kmat, TWmat1, TWmat2;
  Tmat.resize(P.cols(), P.cols());
  Kmat.resize(P.cols(), P.cols());
  TWmat1.resize(P.cols(), P.cols());
  TWmat2.resize(P.cols(), P.cols());
  Eigen::VectorXd unit(P.cols());
  auto idcs = CT.get_indices();
  // generate transformation matrix
  for (auto i = 0; i < P.cols(); ++i) {
    unit.setZero();
    unit(idcs[i]) = 1;
    Tmat.col(i) = ST.sampletTransform(unit);
    for (auto j = 0; j < P.cols(); ++j)
      Kmat(j, i) = exp(-10 * (P.col(idcs[j]) - P.col(idcs[i])).norm());
    TWmat1.col(i) = ST.sampletTransform(Kmat.col(i));
  }
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

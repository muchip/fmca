#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/BlockClusterTree"
#include "FMCA/Samplets"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "print2file.hpp"
#include "util/tictoc.hpp"
#define NPTS 1e4
#define DIM 3

using ClusterT = FMCA::ClusterTree<double, DIM, 3>;

int main() {

  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  // P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  //for (auto i = 0; i < P.cols(); ++i)
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

  return 0;
}
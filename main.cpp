#include <Eigen/Dense>
#include <fstream>

#include "FMCA/ClusterTree"
#include "util/tictoc.hpp"
#define NPTS 1e5
#define DIM 3

void plotBoxes(const std::string& fileName,
               const std::vector<Eigen::Matrix3d>& bb) {
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "# vtk DataFile Version 3.1\n";
  myfile << "this file hopefully represents my surface now\n";
  myfile << "ASCII\n";
  myfile << "DATASET UNSTRUCTURED_GRID\n";
  // print point list
  myfile << "POINTS " << 8 * bb.size() << " FLOAT\n";
  for (auto it = bb.begin(); it != bb.end(); ++it) {
    auto min = it->col(0);
    auto max = it->col(1);
    // lower plane
    myfile << float(min(0)) << " " << float(min(1)) << " " << float(min(2))
           << "\n";
    myfile << float(max(0)) << " " << float(min(1)) << " " << float(min(2))
           << "\n";
    myfile << float(min(0)) << " " << float(max(1)) << " " << float(min(2))
           << "\n";
    myfile << float(max(0)) << " " << float(max(1)) << " " << float(min(2))
           << "\n";
    // upper plane
    myfile << float(min(0)) << " " << float(min(1)) << " " << float(max(2))
           << "\n";
    myfile << float(max(0)) << " " << float(min(1)) << " " << float(max(2))
           << "\n";
    myfile << float(min(0)) << " " << float(max(1)) << " " << float(max(2))
           << "\n";
    myfile << float(max(0)) << " " << float(max(1)) << " " << float(max(2))
           << "\n";
  }
  myfile << "\n";

  // print element list
  myfile << "CELLS " << bb.size() << " " << 9 * bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) {
    myfile << 8;
    for (auto j = 0; j < 8; ++j) myfile << " " << int(8 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) myfile << int(11) << "\n";
  myfile << "\n";

  myfile.close();
  return;
}

template <typename Derived1>
void plotPoints(const std::string& fileName,
                const Eigen::MatrixBase<Derived1>& P) {
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "# vtk DataFile Version 3.1\n";
  myfile << "this file hopefully represents my surface now\n";
  myfile << "ASCII\n";
  myfile << "DATASET UNSTRUCTURED_GRID\n";
  // print point list
  myfile << "POINTS " << P.cols() << " FLOAT\n";
  for (auto i = 0; i < P.cols(); ++i)
    myfile << float(P(0, i)) << " " << float(P(1, i)) << " " << float(P(2, i))
           << "\n";
  myfile << "\n";

  // print element list
  auto nvertices = 1;
  myfile << "CELLS " << P.cols() << " " << (nvertices + 1) * P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i) {
    myfile << int(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << int(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i) myfile << int(1) << "\n";
  myfile << "\n";
  myfile.close();
  return;
}

using ClusterT = FMCA::ClusterTree<double, DIM, 100>;

int main() {
  srand(0);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  //P.row(2) *= 0;
  Eigen::VectorXd nrms = P.colwise().norm();
  for (auto i = 0; i < P.cols(); ++i) P.col(i) *= 1 / nrms(i);
  tictoc T;
  T.tic();
  ClusterT CT(P);
  T.toc("set up ct: ");

  std::vector<std::vector<int>> tree;
  CT.exportTreeStructure(tree);
  for (auto i = 0; i < tree.size(); ++i) {
    int numInd = 0;
    for (auto j = 0; j < tree[i].size(); ++j) numInd += tree[i][j];
    std::cout << i << ") " << tree[i].size() << " " << numInd << "\n";
  }
  std::vector<ClusterT*> leafs;
  CT.getLeafIterator(leafs);
  int numInd = 0;
  for (auto i = 0; i < leafs.size(); ++i)
    numInd += (leafs[i])->get_indices().size();
  std::cout << leafs.size() << " " << numInd << "\n";
  for (auto level = 0; level < 14; ++level) {
    std::vector<Eigen::Matrix3d> bbvec;
    CT.get_BboxVector(&bbvec, level);
    plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
  }
    std::vector<Eigen::Matrix3d> bbvec;
    CT.get_BboxVectorLeafs(&bbvec);
    plotBoxes("boxesLeafs.vtk", bbvec);

  plotPoints("points.vtk", P);
  return 0;
}

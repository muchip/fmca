#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "BoundingBox.hpp"
#include "ClusterTree.hpp"
#include "print2m.h"
#include "tictoc.hpp"
////////////////////////////////////////////////////////////////////////////////
namespace Eigen {
////////////////////////////////////////////////////////////////////////////////
/**
 *    \brief writes a vtk visualization of a given function evaluated
 *           in the surface points P wrt. the mesh topology E
 **/
template <typename Derived1, typename Derived2, typename Derived3>
void surfFunc2vtk(const std::string& fileName,
                  const Eigen::MatrixBase<Derived1>& P,
                  const Eigen::MatrixBase<Derived2>& E,
                  const Eigen::MatrixBase<Derived3>& f) {
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
  auto nvertices = E.rows();
  myfile << "CELLS " << E.cols() << " " << (nvertices + 1) * E.cols() << "\n";
  for (auto i = 0; i < E.cols(); ++i) {
    myfile << int(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << int(E(j, i));
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << E.cols() << "\n";
  for (auto i = 0; i < E.cols(); ++i) myfile << int(9) << "\n";
  myfile << "\n";
  // print point data
  if (f.size() == P.cols()) {
    myfile << "POINT_DATA " << P.cols() << "\n";
    myfile << "SCALARS value FLOAT\n";
    myfile << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < P.cols(); ++i) myfile << float(f(i)) << "\n";
  } else if (f.size() == 3 * P.cols()) {
    myfile << "POINT_DATA " << P.cols() << "\n";
    myfile << "VECTORS gradient FLOAT\n";
    for (auto i = 0; i < P.cols(); ++i)
      myfile << float(f(3 * i)) << " " << float(f(3 * i)) << " "
             << float(f(3 * i)) << "\n";
  }
  myfile.close();
  return;
}

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

}  // namespace Eigen

Eigen::MatrixXi toTriangulation(const Eigen::MatrixXi& E) {
  Eigen::MatrixXi retval(3, 2 * E.cols());
  for (auto i = 0; i < E.cols(); ++i) {
    retval.col(2 * i) << E(0, i), E(1, i), E(2, i);
    retval.col(2 * i + 1) << E(2, i), E(3, i), E(0, i);
  }
  return retval;
}

template <typename T, unsigned int Dim>
Eigen::Matrix<T, Dim, 3u> computeBoundingBox(
    const Eigen::Matrix<T, Dim, Eigen::Dynamic>& P) {
  Eigen::Matrix<T, Dim, 3u> retval;
  retval.col(0) = P.col(0);
  retval.col(1) = P.col(0);
  for (auto i = 1; i < P.cols(); ++i) {
    for (auto j = 0; j < Dim; ++j) {
      // determine minimum
      retval(j, 0) = retval(j, 0) <= P(j, i) ? retval(j, 0) : P(j, i);
      // determine maximum
      retval(j, 1) = retval(j, 1) >= P(j, i) ? retval(j, 1) : P(j, i);
    }
  }
  retval.col(2) = retval.col(1) - retval.col(0);
  return retval;
}

template <typename T, unsigned int Dim>
Eigen::Matrix<T, Dim, 3u> computeBoundingBox2(
    const Eigen::Matrix<T, Dim, Eigen::Dynamic>& P) {
  Eigen::Matrix<T, Dim, 3u> retval;
  for (auto i = 0; i < Dim; ++i) {
    retval(i, 0) = P.row(i).minCoeff();
    retval(i, 1) = P.row(i).maxCoeff();
  }
  retval.col(2) = retval.col(1) - retval.col(0);
  return retval;
}

int main() {
  Eigen::MatrixXi E;
  Eigen::MatrixXd P;

  Eigen::bin2Mat("points.dat", &P);
  std::cout << "points loaded\n";
  Eigen::bin2Mat("elements.dat", &E);
  std::cout << "elements loaded\n";
  surfFunc2vtk("surf.vtk", P, E, Eigen::VectorXd::Zero(P.cols()));
  auto tE = toTriangulation(E);
  surfFunc2vtk("surf2.vtk", P, tE, Eigen::VectorXd::Zero(P.cols()));
  tictoc T;
  T.tic();
  BoundingBox<double, 3> bb(P);
  T.toc("time for bb: ");
  T.tic();
  ClusterTree<double, 3, 1, 4> ct(P);
  T.toc("time for clusterTree: ");
  ct.shrinkToFit(P);
  std::cout << bb.get_bb() << std::endl;
  std::vector<Eigen::Matrix3d> bbvec;
  ct.get_BboxVector(&bbvec);
  // bbvec.push_back(bb.get_bb());
  // bbvec.push_back(bb.split(1, 0).get_bb());
  // bbvec.push_back(bb.split(1, 1).get_bb());
  plotBoxes("boxes.vtk", bbvec);
  return 0;
}

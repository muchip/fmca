// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_IO_H_
#define FMCA_UTIL_IO_H_

#include <fstream>

namespace FMCA {
namespace IO {
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief exports a sequence of 3D boxes stored in a std::vector in vtk
 **/
void plotBoxes(const std::string &fileName,
               const std::vector<Eigen::MatrixXd> &bb) {
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
    for (auto j = 0; j < 8; ++j)
      myfile << " " << int(8 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i)
    myfile << int(11) << "\n";
  myfile << "\n";

  myfile.close();
  return;
}

/**
 *  \brief exports a sequence of 3D boxes stored in a std::vector in vtk
 **/
template <typename Bbvec>
void plotBoxes2D(const std::string &fileName, Bbvec &bb) {
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "# vtk DataFile Version 3.1\n";
  myfile << "this file hopefully represents my surface now\n";
  myfile << "ASCII\n";
  myfile << "DATASET UNSTRUCTURED_GRID\n";
  // print point list
  myfile << "POINTS " << 4 * bb.size() << " FLOAT\n";
  for (auto it = bb.begin(); it != bb.end(); ++it) {
    auto min = it->col(0);
    auto max = it->col(1);
    // lower plane
    myfile << float(min(0)) << " " << float(min(1)) << " " << 0 << "\n";
    myfile << float(max(0)) << " " << float(min(1)) << " " << 0 << "\n";
    myfile << float(min(0)) << " " << float(max(1)) << " " << 0 << "\n";
    myfile << float(max(0)) << " " << float(max(1)) << " " << 0 << "\n";
  }
  myfile << "\n";

  // print element list
  myfile << "CELLS " << bb.size() << " " << 5 * bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) {
    myfile << 4;
    for (auto j = 0; j < 4; ++j)
      myfile << " " << int(4 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i)
    myfile << int(8) << "\n";
  myfile << "\n";

  myfile.close();
  return;
}

////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief exports a sequence of 3D boxes stored in a std::vector in vtk
 **/
template <typename Scalar>
void plotBoxes(const std::string &fileName,
               const std::vector<Eigen::Matrix3d> &bb,
               const std::vector<Scalar> &cvec) {
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
    for (auto j = 0; j < 8; ++j)
      myfile << " " << int(8 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i)
    myfile << int(11) << "\n";
  myfile << "\n";
  myfile << "CELL_DATA " << cvec.size() << "\n";
  myfile << "SCALARS coefficients FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < cvec.size(); ++i)
    myfile << cvec[i] << "\n";
  myfile.close();
  return;
}
/**
 *  \brief exports a list of points in vtk
 **/
template <typename Derived>
void plotPoints(const std::string &fileName,
                const Eigen::MatrixBase<Derived> &P) {
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
    for (auto j = 0; j < nvertices; ++j)
      myfile << " " << int(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i)
    myfile << int(1) << "\n";
  myfile << "\n";
  myfile.close();
  return;
}
/**
 *  \brief exports a list of points in vtk
 **/
template <typename ClusterTree, typename Derived, typename otherDerived>
void plotPoints(const std::string &fileName, const ClusterTree &CT,
                const Eigen::MatrixBase<Derived> &P,
                const Eigen::MatrixBase<otherDerived> &fdat) {
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "# vtk DataFile Version 3.1\n";
  myfile << "this file hopefully represents my surface now\n";
  myfile << "ASCII\n";
  myfile << "DATASET UNSTRUCTURED_GRID\n";
  // print point list
  auto idcs = CT.indices();
  myfile << "POINTS " << P.cols() << " FLOAT\n";
  for (auto i = 0; i < P.cols(); ++i)
    myfile << float(P(0, idcs[i])) << " " << float(P(1, idcs[i])) << " "
           << float(P(2, idcs[i])) << "\n";
  myfile << "\n";

  // print element list
  auto nvertices = 1;
  myfile << "CELLS " << P.cols() << " " << (nvertices + 1) * P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i) {
    myfile << int(nvertices);
    for (auto j = 0; j < nvertices; ++j)
      myfile << " " << int(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i)
    myfile << int(1) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "POINT_DATA " << fdat.size() << "\n";
  myfile << "SCALARS f_values FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i)
    myfile << fdat(i) << "\n";
  myfile.close();

  return;
}

/**
 *  \brief exports a list of points in vtk
 **/
template <typename Derived, typename otherDerived>
void plotPointsColor(const std::string &fileName,
                     const Eigen::MatrixBase<Derived> &P,
                     const Eigen::MatrixBase<otherDerived> &fdat) {
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
    for (auto j = 0; j < nvertices; ++j)
      myfile << " " << int(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i)
    myfile << int(1) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "POINT_DATA " << fdat.size() << "\n";
  myfile << "SCALARS f_values FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i)
    myfile << fdat(i) << "\n";
  myfile.close();

  return;
}

/**
 *  \brief exports a list of points in vtk
 **/
template <typename Derived1, typename Derived2, typename Derived3>
void plotTriMeshColor(const std::string &fileName,
                      const Eigen::MatrixBase<Derived1> &P,
                      const Eigen::MatrixBase<Derived2> &F,
                      const Eigen::MatrixBase<Derived3> &fdat) {
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
  auto nvertices = 3;
  myfile << "CELLS " << F.rows() << " " << (nvertices + 1) * F.rows() << "\n";
  for (auto i = 0; i < F.rows(); ++i) {
    myfile << int(nvertices);
    for (auto j = 0; j < nvertices; ++j)
      myfile << " " << int(F(i, j));
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << F.rows() << "\n";
  for (auto i = 0; i < F.rows(); ++i)
    myfile << int(5) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "POINT_DATA " << fdat.size() << "\n";
  myfile << "SCALARS f_values FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i)
    myfile << fdat(i) << "\n";
  myfile.close();

  return;
}

/**
 *  \brief exports a list of points in vtk
 **/
template <typename Derived1, typename Derived2, typename Derived3>
void plotTriMeshColor2(const std::string &fileName,
                      const Eigen::MatrixBase<Derived1> &P,
                      const Eigen::MatrixBase<Derived2> &F,
                      const Eigen::MatrixBase<Derived3> &fdat) {
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
  auto nvertices = 3;
  myfile << "CELLS " << F.rows() << " " << (nvertices + 1) * F.rows() << "\n";
  for (auto i = 0; i < F.rows(); ++i) {
    myfile << int(nvertices);
    for (auto j = 0; j < nvertices; ++j)
      myfile << " " << int(F(i, j));
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << F.rows() << "\n";
  for (auto i = 0; i < F.rows(); ++i)
    myfile << int(5) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "CELL_DATA " << fdat.size() << "\n";
  myfile << "SCALARS rho FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i)
    myfile << fdat(i) << "\n";
  myfile.close();

  return;
}

} // namespace IO
} // namespace FMCA
#endif

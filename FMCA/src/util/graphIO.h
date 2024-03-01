// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_GRAPHIO_H_
#define FMCA_UTIL_GRAPHIO_H_

#include <Eigen/Sparse>
#include <fstream>

#include "Macros.h"

namespace FMCA {
namespace IO {
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief exports a list of points in vtk
 **/
template <typename Derived>
void plotGraph(const std::string &fileName, const Eigen::MatrixBase<Derived> &P,
               const std::vector<Eigen::Triplet<Index>> &A) {
  if (P.rows() > 3 || P.rows() < 2) return;
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "# vtk DataFile Version 3.1\n";
  myfile << "this file hopefully represents my surface now\n";
  myfile << "ASCII\n";
  myfile << "DATASET UNSTRUCTURED_GRID\n";
  // print point list
  myfile << "POINTS " << P.cols() << " FLOAT\n";
  for (auto i = 0; i < P.cols(); ++i) {
    float thirdC = 0;
    if (P.rows() == 3) thirdC = P(2, i);

    myfile << float(P(0, i)) << " " << float(P(1, i)) << " " << thirdC << "\n";
  }
  myfile << "\n";

  // print element list
  const int nvertices = 2;
  const int nedges = A.size();
  myfile << "CELLS " << nedges << " " << (nvertices + 1) * nedges << "\n";
  for (auto i = 0; i < A.size(); ++i) {
    myfile << Index(nvertices);
    myfile << " " << A[i].row() << " " << A[i].col() << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << nedges << "\n";
  for (auto i = 0; i < nedges; ++i) myfile << Index(3) << "\n";
  myfile << "\n";
  myfile.close();
  return;
}

}  // namespace IO
}  // namespace FMCA
#endif

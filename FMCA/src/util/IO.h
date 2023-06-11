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
#ifndef FMCA_UTIL_IO_H_
#define FMCA_UTIL_IO_H_

#include <fstream>
#include <iomanip>
#include <iostream>

#include "./Macros.h"

namespace FMCA {
namespace IO {
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief exports a sequence of 3D boxes stored in a std::vector in vtk
 **/
void plotBoxes(const std::string &fileName, const std::vector<Matrix> &bb) {
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
    for (auto j = 0; j < 8; ++j) myfile << " " << Index(8 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) myfile << Index(11) << "\n";
  myfile << "\n";

  myfile.close();
  return;
}

/**
 *  \brief exports a sequence of 3D boxes stored in a std::vector in vtk
 **/
template <typename Bbvec>
void plotBoxes2D(const std::string &fileName, Bbvec &bb,
                 const std::vector<Scalar> &cvec) {
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
    for (auto j = 0; j < 4; ++j) myfile << " " << Index(4 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) myfile << Index(8) << "\n";
  myfile << "\n";
  myfile << "CELL_DATA " << cvec.size() << "\n";
  myfile << "SCALARS coefficients FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < cvec.size(); ++i) myfile << cvec[i] << "\n";
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
    for (auto j = 0; j < 4; ++j) myfile << " " << Index(4 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) myfile << Index(8) << "\n";
  myfile << "\n";

  myfile.close();
  return;
}

////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief exports a sequence of 3D boxes stored in a std::vector in vtk
 **/
template <typename Scalar>
void plotBoxes(const std::string &fileName, const std::vector<Matrix> &bb,
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
    for (auto j = 0; j < 8; ++j) myfile << " " << Index(8 * i + j);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << bb.size() << "\n";
  for (auto i = 0; i < bb.size(); ++i) myfile << Index(11) << "\n";
  myfile << "\n";
  myfile << "CELL_DATA " << cvec.size() << "\n";
  myfile << "SCALARS coefficients FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < cvec.size(); ++i) myfile << cvec[i] << "\n";
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
    myfile << Index(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << Index(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i) myfile << Index(1) << "\n";
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
    myfile << Index(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << Index(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i) myfile << Index(1) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "POINT_DATA " << fdat.size() << "\n";
  myfile << "SCALARS f_values FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i) myfile << fdat(i) << "\n";
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
    myfile << Index(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << Index(i);
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << P.cols() << "\n";
  for (auto i = 0; i < P.cols(); ++i) myfile << Index(1) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "POINT_DATA " << fdat.size() << "\n";
  myfile << "SCALARS f_values FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i) myfile << fdat(i) << "\n";
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
    myfile << Index(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << Index(F(i, j));
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << F.rows() << "\n";
  for (auto i = 0; i < F.rows(); ++i) myfile << Index(5) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "POINT_DATA " << fdat.size() << "\n";
  myfile << "SCALARS f_values FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i) myfile << fdat(i) << "\n";
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
    myfile << Index(nvertices);
    for (auto j = 0; j < nvertices; ++j) myfile << " " << Index(F(i, j));
    myfile << "\n";
  }
  myfile << "\n";

  myfile << "CELL_TYPES " << F.rows() << "\n";
  for (auto i = 0; i < F.rows(); ++i) myfile << Index(5) << "\n";
  myfile << "\n";

  /* print z-values of the geometry and solved density for visualization */
  myfile << "CELL_DATA " << fdat.size() << "\n";
  myfile << "SCALARS rho FLOAT\n";
  myfile << "LOOKUP_TABLE default\n";
  for (auto i = 0; i < fdat.size(); ++i) myfile << fdat(i) << "\n";
  myfile.close();

  return;
}

/**
 *  \brief write Eigen::Matrix into an ascii txt file.
 **/
template <typename Derived>
int print2ascii(const std::string &fileName,
                const Eigen::MatrixBase<Derived> &var) {
  // evaluate Eigen expression into a matrix
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> tmp =
      var;

  std::ofstream myfile;

  myfile.open(fileName);
  // write matrix to file (precision is fixed here!)
  for (auto i = 0; i < tmp.rows(); ++i) {
    for (auto j = 0; j < tmp.cols(); ++j)
      myfile << std::setprecision(10) << tmp(i, j) << " \t ";
    myfile << std::endl;
  }
  myfile.close();

  return 0;
}

template <typename SparseMatrix>
int print2spascii(const std::string &fileName, const SparseMatrix &var,
                  const std::string &writeMode) {
  std::ofstream myfile;
  if (writeMode == "w")
    myfile.open(fileName);
  else if (writeMode == "a")
    myfile.open(fileName, std::ios_base::app);
  else
    return 1;
  for (auto i = 0; i < var.outerSize(); i++)
    for (typename SparseMatrix::InnerIterator it(var, i); it; ++it) {
      myfile << it.row() + 1 << " " << it.col() + 1 << " "
             << std::setprecision(10) << it.value() << std::endl;
    }
  myfile.close();
  return 0;
}

/**
 *  \brief write Eigen::Matrix into a Matlab .m file.
 **/
template <typename Derived>
int print2m(const std::string &fileName, const std::string &varName,
            const Eigen::MatrixBase<Derived> &var,
            const std::string &writeMode) {
  // evaluate Eigen expression into a matrix
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> tmp =
      var;

  std::ofstream myfile;
  // if flag is set to w, a new file is created, otherwise the new matrix
  // is just appended
  if (writeMode == "w")
    myfile.open(fileName);
  else if (writeMode == "a")
    myfile.open(fileName, std::ios_base::app);
  else
    return 1;

  myfile << varName << "=[" << std::endl;

  for (int i = 0; i < (int)tmp.rows(); ++i) {
    for (int j = 0; j < (int)tmp.cols(); ++j)
      myfile << std::setprecision(10) << tmp(i, j) << " \t ";
    myfile << std::endl;
  }
  myfile << "];" << std::endl;

  myfile.close();

  return 0;
}

template <typename Scalar>
int print2m(const std::string &fileName, const std::string &varName,
            const Eigen::SparseMatrix<Scalar> &var,
            const std::string &writeMode) {
  Eigen::VectorXd rowInd(var.nonZeros());
  Eigen::VectorXd colInd(var.nonZeros());
  Eigen::VectorXd value(var.nonZeros());
  unsigned int j = 0;
  for (auto i = 0; i < var.outerSize(); i++)
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(var, i); it;
         ++it) {
      rowInd(j) = it.row() + 1;
      colInd(j) = it.col() + 1;
      value(j) = it.value();
      ++j;
    }
  print2m(fileName, "rows_" + varName, rowInd, writeMode);
  print2m(fileName, "cols_" + varName, colInd, "a");
  print2m(fileName, "values_" + varName, value, "a");
  std::ofstream myfile;
  // if flag is set to w, a new file is created, otherwise the new matrix
  // is just appended
  myfile.open(fileName, std::ios_base::app);
  myfile << varName << " = sparse("
         << "rows_" + varName << ","
         << "cols_" + varName << ","
         << "values_" + varName << "," << var.rows() << "," << var.cols()
         << ");\n";
  myfile.close();

  return 0;
}

template <typename Derived>
int print2bin(const std::string &fileName,
              const Eigen::MatrixBase<Derived> &var) {
  typedef typename Derived::Scalar DataType;
  std::ofstream myfile;
  size_t rows = 0;
  size_t cols = 0;
  size_t IsRowMajor = 0;
  size_t dataSize = 0;
  DataType *data = nullptr;

  IsRowMajor = var.IsRowMajor;
  dataSize = sizeof(DataType);

  myfile.open(fileName, std::ios::out | std::ios::binary);
  // write data size and row major flag
  myfile.write(reinterpret_cast<const char *>(&dataSize), sizeof(size_t));
  myfile.write(reinterpret_cast<const char *>(&IsRowMajor), sizeof(size_t));
  rows = var.rows();
  cols = var.cols();
  // write rows and cols of the matrix
  myfile.write((const char *)&(rows), sizeof(size_t));
  myfile.write((const char *)&(cols), sizeof(size_t));
  std::cout << "Eigen to binary file writer" << std::endl;
  std::cout << "rows: " << rows << " cols: " << cols
            << " dataSize: " << dataSize << " IsRowMajor: " << IsRowMajor
            << std::endl;
  std::cout << "writing..." << std::flush;
  const DataType *data_ptr = var.derived().data();
  if (IsRowMajor) {
    data = new (std::nothrow) DataType[cols];
    assert(data != nullptr && "allocation failed");
    for (auto i = 0; i < rows; ++i) {
      memcpy(data, data_ptr + i * cols, cols * sizeof(DataType));
      myfile.write((const char *)data, size_t(cols) * sizeof(DataType));
    }

  } else {
    data = new (std::nothrow) DataType[rows];
    assert(data != nullptr && "allocation failed");
    for (auto i = 0; i < cols; ++i) {
      memcpy(data, data_ptr + i * rows, rows * sizeof(DataType));
      myfile.write((const char *)data, size_t(rows) * sizeof(DataType));
    }
  }
  std::cout << " done." << std::endl;
  std::cout << rows / 1000. * cols / 1000. * sizeof(DataType) / 1000.
            << "GB written to file" << std::endl;
  myfile.close();
  delete[] data;
  return 0;
}

template <typename Derived>
int bin2Mat(const std::string &fileName,
            Eigen::MatrixBase<Derived> *targetMat) {
  typedef typename Derived::Scalar DataType;
  std::ifstream myfile;
  size_t rows = 0;
  size_t cols = 0;
  size_t IsRowMajor = 0;
  size_t dataSize = 0;
  DataType *data = nullptr;

  myfile.open(fileName, std::ios::in | std::ios::binary);

  myfile.read(reinterpret_cast<char *>(&dataSize), sizeof(size_t));
  myfile.read(reinterpret_cast<char *>(&IsRowMajor), sizeof(size_t));
  myfile.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
  myfile.read(reinterpret_cast<char *>(&cols), sizeof(size_t));
  std::cout << "Binary file to Eigen reader" << std::endl;
  std::cout << "rows: " << rows << " cols: " << cols
            << " dataSize: " << dataSize << " IsRowMajor: " << IsRowMajor
            << std::endl;
  if (dataSize != sizeof(DataType)) {
    std::cout << "mismatch in data size of target and input file size"
              << std::endl;
    return 1;
  }

  Derived &ret_val = targetMat->derived();
  ret_val.resize(rows, cols);
  std::cout << "reading..." << std::flush;
  if (IsRowMajor) {
    data = new (std::nothrow) DataType[cols];
    assert(data != nullptr && "allocation failed");
    for (auto i = 0; i < rows; ++i) {
      myfile.read((char *)data, cols * sizeof(DataType));
      ret_val.row(i) =
          Eigen::Map<Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>(data, 1, cols);
    }
  } else {
    data = new (std::nothrow) DataType[rows];
    assert(data != nullptr && "allocation failed");
    for (auto i = 0; i < cols; ++i) {
      myfile.read((char *)data, rows * sizeof(DataType));
      ret_val.col(i) =
          Eigen::Map<Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor>>(data, rows, 1);
    }
  }
  std::cout << " done." << std::endl;
  std::cout << rows / 1000. * cols / 1000. * sizeof(DataType) / 1000.
            << "GB read from file" << std::endl;
  myfile.close();
  delete[] data;
  return 0;
}

Matrix ascii2Matrix(const std::string &filename) {
  std::cout << "loading " << filename << std::flush;
  int cols = 0;
  int rows = 0;
  std::vector<double> buff;
  buff.resize(int(1e8));
  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(filename);
  while (!infile.eof()) {
    std::string line;
    std::getline(infile, line);
    int temp_cols = 0;
    std::stringstream stream(line);
    while (!stream.eof()) stream >> buff[cols * rows + temp_cols++];

    if (temp_cols == 0) continue;

    if (cols == 0) cols = temp_cols;

    ++rows;
  }

  infile.close();

  --rows;

  // Populate matrix with numbers.
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) result(i, j) = buff[cols * i + j];
  std::cout << " done.\n" << std::flush;
  std::cout << "data size: " << rows << " x " << cols << std::endl
            << std::flush;
  return result;
};

}  // namespace IO
}  // namespace FMCA
#endif

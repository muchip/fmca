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
#ifndef FMCA_UTIL_IO_PRINT2FILE_H_
#define FMCA_UTIL_IO_PRINT2FILE_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace FMCA {
namespace IO {

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

template <typename Scalar>
int print2spascii(const std::string &fileName,
                  const Eigen::SparseMatrix<Scalar> &var,
                  const std::string &writeMode) {
  std::ofstream myfile;
  if (writeMode == "w")
    myfile.open(fileName);
  else if (writeMode == "a")
    myfile.open(fileName, std::ios_base::app);
  else
    return 1;
  for (auto i = 0; i < var.outerSize(); i++)
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(var, i); it;
         ++it) {
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
}  // namespace IO
}  // namespace FMCA

#endif

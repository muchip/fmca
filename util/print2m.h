#ifndef __PRINT2M__
#define __PRINT2M__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Eigen/Core"

namespace Eigen {

template <typename Derived>
int print2ascii(const std::string &fileName,
                const Eigen::MatrixBase<Derived> &var) {
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> tmp =
      var;

  std::ofstream myfile;

  myfile.open(fileName);

  for (int i = 0; i < (int)tmp.rows(); ++i) {
    for (int j = 0; j < (int)tmp.cols(); ++j)
      myfile << std::setprecision(10) << tmp(i, j) << " \t ";
    myfile << std::endl;
  }
  myfile.close();

  return 0;
}

template <typename Derived>
int print2m(const std::string &fileName, const std::string &varName,
            const Eigen::MatrixBase<Derived> &var,
            const std::string &writeMode) {
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> tmp =
      var;

  std::ofstream myfile;

  if (writeMode == "w")
    myfile.open(fileName);
  else if (writeMode == "a")
    myfile.open(fileName, std::ios_base::app);
  else
    return 1;

  myfile << varName << "=[" << std::endl;

  for (int i = 0; i < (int)tmp.rows(); ++i) {
    for (int j = 0; j < (int)tmp.cols(); ++j)
      myfile << std::setprecision(30) << tmp(i, j) << " \t ";
    myfile << std::endl;
  }
  myfile << "];" << std::endl;

  myfile.close();

  return 0;
}

template <typename Derived>
int print2bin(const std::string &fileName,
              const Eigen::MatrixBase<Derived> &var) {
  std::ofstream myfile;
  int rows = 0;
  int cols = 0;
  int IsRowMajor = 0;
  int dataSize = 0;

  IsRowMajor = var.IsRowMajor;
  dataSize = sizeof(typename Derived::Scalar);

  myfile.open(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

  myfile.write((const char *)&(dataSize), sizeof(int));
  myfile.write((const char *)&(IsRowMajor), sizeof(int));
  if (IsRowMajor) {
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::RowMajor>
        tmp = var;

    rows = tmp.rows();
    cols = tmp.cols();
    std::cout << dataSize << " " << IsRowMajor << " " << rows << " " << cols
              << std::endl;

    myfile.write((const char *)&(rows), sizeof(int));
    myfile.write((const char *)&(cols), sizeof(int));
    myfile.write((const char *)tmp.data(), rows * cols * dataSize);

  } else {
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::ColMajor>
        tmp = var;

    rows = tmp.rows();
    cols = tmp.cols();
    std::cout << dataSize << " " << IsRowMajor << " " << rows << " " << cols
              << std::endl;

    myfile.write((const char *)&(rows), sizeof(int));
    myfile.write((const char *)&(cols), sizeof(int));

    myfile.write((const char *)tmp.data(), rows * cols * dataSize);
  }

  myfile.close();

  return 0;
}

template <typename Derived>
int bin2Mat(const std::string &fileName,
            Eigen::MatrixBase<Derived> *targetMat) {
  std::ifstream myfile;
  int rows = 0;
  int cols = 0;
  int IsRowMajor = 0;
  int dataSize = 0;
  typename Derived::Scalar *data = NULL;
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
      returnMat;

  myfile.open(fileName, std::ios::in | std::ios::binary);

  myfile.read((char *)&(dataSize), sizeof(int));
  myfile.read((char *)&(IsRowMajor), sizeof(int));
  myfile.read((char *)&(rows), sizeof(int));
  myfile.read((char *)&(cols), sizeof(int));
  std::cout << dataSize << " " << IsRowMajor << " " << rows << " " << cols
            << std::endl;
  if (dataSize != sizeof(typename Derived::Scalar)) {
    std::cout << "mismatch in data size of target and input file size"
              << "(file: " << dataSize
              << " mat: " << sizeof(typename Derived::Scalar) << ")"
              << std::endl;
    return 1;
  }

  data = new typename Derived::Scalar[rows * cols];

  myfile.read((char *)data, rows * cols * dataSize);

  myfile.close();

  if (IsRowMajor)
    returnMat = Map<Matrix<typename Derived::Scalar, Eigen::Dynamic,
                           Eigen::Dynamic, Eigen::RowMajor> >(data, rows, cols);
  else
    returnMat = Map<Matrix<typename Derived::Scalar, Eigen::Dynamic,
                           Eigen::Dynamic, Eigen::ColMajor> >(data, rows, cols);

  *targetMat = returnMat;

  delete[] data;
  return 0;
}
}  // namespace Eigen

#endif

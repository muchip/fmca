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
#ifndef FMCA_POINTS_MATRIXREADER_H_
#define FMCA_POINTS_MATRIXREADER_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

Eigen::MatrixXd readMatrix(const std::string &filename) {
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

    rows++;
  }

  infile.close();

  rows--;

  // Populate matrix with numbers.
  Eigen::MatrixXd result(rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) result(i, j) = buff[cols * i + j];
  std::cout << " done.\n" << std::flush;
  std::cout << "data size: " << rows << " x " << cols << std::endl << std::flush;
  return result;
};

#endif

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

Eigen::MatrixXd readMatrix(const std::string &filename) {
  std::cout << filename << std::endl;
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
    while (!stream.eof())
      stream >> buff[cols * rows + temp_cols++];

    if (temp_cols == 0)
      continue;

    if (cols == 0)
      cols = temp_cols;

    rows++;
  }

  infile.close();

  rows--;

  // Populate matrix with numbers.
  Eigen::MatrixXd result(rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      result(i, j) = buff[cols * i + j];

  return result;
};

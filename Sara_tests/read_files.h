#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "FMCA/CovarianceKernel"
#include "FMCA/Samplets"
#include "FMCA/src/util/Tictoc.h"


#define MPOLE_DEG 6

void readCSV(const std::string &filename, FMCA::Matrix &matrix, int &npts, const int dim) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<std::vector<double>> data;
    std::getline(file, line); // no header line

    while (std::getline(file, line)) {
        std::istringstream linestream(line);
        std::vector<double> row;
        std::getline(linestream, cell, ','); // skip index
        while (std::getline(linestream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    npts = data.size();
    matrix.resize(dim, npts); // Transpose: swap dimensions
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < npts; ++j) {
            matrix(i, j) = data[j][i]; // Transpose: swap indices
        }
    }
}

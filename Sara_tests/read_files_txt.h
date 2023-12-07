#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "FMCA/CovarianceKernel"
#include "FMCA/Samplets"
#include "FMCA/src/util/Tictoc.h"


void readTXT(const std::string &filename, FMCA::Matrix &matrix, int &npts, const int dim) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        std::istringstream linestream(line);
        std::vector<double> row;

        for (int i = 0; i < dim; ++i) {
            double value;
            if (linestream >> value) {
                row.push_back(value);
            } else {
                // Handle the case where the line doesn't have enough values
                std::cerr << "Error: Insufficient values in the line." << std::endl;
                exit(EXIT_FAILURE);
            }
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

// int main() {
//     // Example usage
//     FMCA::Matrix matrix;
//     int npts;
//     const int dim = 2; // Change this to the actual dimension of your data

//     readTXT("baricenters_circle.txt", matrix, npts, dim);
//     std::cout << matrix << std::endl;

    
//     return 0;
// }

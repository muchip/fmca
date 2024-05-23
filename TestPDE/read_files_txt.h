#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "FMCA/CovarianceKernel"
#include "FMCA/Samplets"
#include "FMCA/src/util/Tictoc.h"

void readTXT(const std::string &filename, FMCA::Matrix &matrix, const int dim) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::vector<double> temp_values;
    while (std::getline(file, line)) {
        std::istringstream linestream(line);
        double value;
        while (linestream >> value) {
            temp_values.push_back(value);
        }
    }

    int npts = temp_values.size() / dim;
    matrix.resize(dim, npts);

    for (size_t i = 0; i < temp_values.size(); ++i) {
        size_t row = i % dim;      // Determine the row index
        size_t col = i / dim;      // Determine the column index
        matrix(row, col) = temp_values[i];
    }
}



// Overloaded function for handling vectors
void readTXT(const std::string &filename, FMCA::Vector &vector) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::vector<double> temp_values;
    while (std::getline(file, line)) {
        std::istringstream linestream(line);
        double value;
        if (linestream >> value) {
            temp_values.push_back(value);
        } else {
            std::cerr << "Error: Insufficient values in the line." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int npts = temp_values.size();
    vector.resize(npts);
    for (int i = 0; i < npts; ++i) {
        vector(i) = temp_values[i];
    }
}



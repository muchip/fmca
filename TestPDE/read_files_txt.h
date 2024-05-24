/* This file reads txt imput of data. The input should be a matrix of coordinates structured like this:
x1 y1
x2 y2
x3 y3
...
Since FMCA library relies on row major data, so the fucntions below reads the common column files coordinates
and return the followiing format
x1 x3 x3 ...
y1 y2 y3 ... */


#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

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



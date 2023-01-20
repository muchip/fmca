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

#include "../FMCA/src/util/Macros.h"

namespace FMCA {
namespace IO_CSV {
FMCA::Matrix readCSV(const std::string &fileName) {
  std::ofstream myfileStream;
  std::ifstream myfileStream(fileName);
    if (myfileStream.fail()) {
        throw std::runtime_error("File is not found");
    }
    std::string line;
    std::vector<std::vector<std::string>> matrixContent;
    // We loop in the file line by line
    while (std::getline(myfileStream, line)) {
        std::stringstream lineStream(line);
        std::vector<std::string> tokens;
        std::string token;
        std::char delimiter = ',';
        // how the line is terminate 
        while (std::getline(lineStream, token,delimiter)) {
            tokens.push_back(token);
        }
        matrixContent.emplace_back(tokens);
    }  
    myfileStream.close();
    size_t numberOfRows = hasColumnAndRowNames ? matrixContent.size() - 1
                                               : matrixContent.size();
    size_t numberOfColumns = hasColumnAndRowNames ? matrixContent.at(0).size() - 1
                                                  : matrixContent.at(0).size();
    
    

    FMCA::Matrix result.resize(numberOfRows, numberOfColumns);


    for (size_t i = 0; i < numberOfRows; ++i) {
        for (size_t j = 0; j < numberOfColumns; ++j) {
            size_t row = hasColumnAndRowNames ? i + 1 : i;
            size_t col = hasColumnAndRowNames ? j + 1 : j;
            FMCA::Scalar value = std::stod(matrixContent.at(row).at(col));
            if (value != 0) {
                resuls(i,j)=value;
            }
        }
    }
    return result;
 }
} //End IO
} // end FMCA
#endif
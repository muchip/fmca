#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include "InputOutput.h"
#include "Regression.h"
#include "SmoothnessDetection.h"

#define DIM 1

using ST = FMCA::SampletTree<FMCA::ClusterTree>;

using namespace FMCA;

void promptUserInput(const std::string& prompt, std::string& value, const std::string& defaultValue) {
    std::cout << prompt << " (default: " << defaultValue << "): ";
    std::getline(std::cin, value);
    if (value.empty()) {
        value = defaultValue;
    }
}

void promptUserInput(const std::string& prompt, Scalar& value, Scalar defaultValue) {
    std::cout << prompt << " (default: " << defaultValue << "): ";
    std::string input;
    std::getline(std::cin, input);
    if (!input.empty()) {
        value = std::stod(input);
    } else {
        value = defaultValue;
    }
}

void promptUserInput(const std::string& prompt, Index& value, Index defaultValue) {
    std::cout << prompt << " (default: " << defaultValue << "): ";
    std::string input;
    std::getline(std::cin, input);
    if (!input.empty()) {
        value = std::stoi(input);
    } else {
        value = defaultValue;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
    // Default parameters
    std::string inputCoordinates = "coordinates.txt";
    std::string inputValues = "values.txt";
    std::string outputFileSteps = "output_step.txt";
    Index dtilde = 10;

    // User input
    promptUserInput("Enter the file name for coordinates", inputCoordinates, inputCoordinates);
    promptUserInput("Enter the file name for values", inputValues, inputValues);
    promptUserInput("Enter the output file name for steps plot", outputFileSteps, outputFileSteps);
    promptUserInput("Insert the number of vanishing moments dtilde (chose here a high dtilde for the smoothness class detection)", dtilde, dtilde);

    // Display chosen parameters
    std::cout << "\nUsing the following parameters:\n";
    std::cout << "Input coordinates file: " << inputCoordinates << "\n";
    std::cout << "Input values file: " << inputValues << "\n";
    std::cout << "Output file for steps plot: " << outputFileSteps << "\n";
    std::cout << "Vanishing moments (dtilde): " << dtilde << "\n";

    /////////////////////////////////
    Matrix P;
    readTXT(inputCoordinates, P, DIM);
    std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

    Vector f;
    readTXT(inputValues, f);
    std::cout << "Dimension f = " << f.rows() << " x " << f.cols() << std::endl;

    /////////////////////////////////
    const Scalar eta = 1. / DIM;
    const Scalar mpoleDeg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
    std::cout << "eta                 " << eta << std::endl;
    std::cout << "dtilde              " << dtilde << std::endl;
    std::cout << "thresholdKernel     " << thresholdKernel << std::endl;
    std::cout << "mpoleDeg            " << mpoleDeg << std::endl;

    const Moments mom(P, mpoleDeg);
    const SampletMoments sampMom(P, dtilde - 1);
    const ST hst(sampMom, 0, P);
    std::cout << "Tree done." << std::endl;

    Vector f_ordered = hst.toClusterOrder(f);
    Vector f_samplets = hst.sampletTransform(f_ordered);

    /////////////////////////////////////////////////
    Vector tdata = f_samplets;

     Index max_level = computeMaxLevel(hst);
    std::cout << "Maximum level of the tree: " << max_level << std::endl;
    const Index nclusters = std::distance(hst.begin(), hst.end());
    std::cout << "Total number of clusters: " << nclusters << std::endl;

    std::map<const ST*, std::vector<Scalar>> leafCoefficients;
    traverseAndStackCoefficients(hst, tdata, leafCoefficients);

    const std::string outputFile_slopes = "relativeSlopes.csv";
    auto slopes = computeRelativeSlopes1D<ST, Scalar>(leafCoefficients, dtilde, 1e-4,
                                                        outputFile_slopes);
    // auto slopes = computeLinearRegressionSlope(leafCoefficients, dtilde);


    generateStepFunctionData(slopes, outputFile_step);

    // Coeff decay
    printMaxCoefficientsPerLevel(hst, tdata);

    return 0;
}

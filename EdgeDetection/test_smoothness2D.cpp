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

#define DIM 2

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
    std::string outputFileBoxes = "output_boxes.py";
    Scalar thresholdActiveLeaves = 1e-6;
    Index dtilde = 4;

    // User input
    promptUserInput("Enter the file name for coordinates", inputCoordinates, inputCoordinates);
    promptUserInput("Enter the file name for values", inputValues, inputValues);
    promptUserInput("Enter the output file name for bounding boxes", outputFileBoxes, outputFileBoxes);
    promptUserInput("Insert the number of vanishing moments dtilde (higher dtilde, bigger bounding boxes)", dtilde, dtilde);
    promptUserInput("Insert the threshold for active leaves (higher threshold, less active leaves)", thresholdActiveLeaves, thresholdActiveLeaves);

    // Display chosen parameters
    std::cout << "\nUsing the following parameters:\n";
    std::cout << "Input coordinates file: " << inputCoordinates << "\n";
    std::cout << "Input values file: " << inputValues << "\n";
    std::cout << "Output file for bounding boxes: " << outputFileBoxes << "\n";
    std::cout << "Vanishing moments (dtilde): " << dtilde << "\n";
    std::cout << "Threshold for active leaves: " << thresholdActiveLeaves << "\n";

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
    std::cout << "mpoleDeg            " << mpoleDeg << std::endl;

    const Moments mom(P, mpoleDeg);
    const SampletMoments sampMom(P, dtilde - 1);
    const ST hst(sampMom, 0, P);
    std::cout << "Tree done." << std::endl;

    Vector f_ordered = hst.toClusterOrder(f);
    Vector f_samplets = hst.sampletTransform(f_ordered);

    /////////////////////////////////////////////////
    Vector tdata = f_samplets;

    Index maxLevel = computeMaxLevel(hst);
    std::cout << "Maximum level of the tree: " << maxLevel << std::endl;
    const Index nClusters = std::distance(hst.begin(), hst.end());
    std::cout << "Total number of clusters: " << nClusters << std::endl;

    printMaxCoefficientsPerLevel(hst, tdata);

    // Detect singularities
    std::vector<const ST*> adaptiveTree =
        computeAdaptiveTree(hst, tdata, thresholdActiveLeaves);
    std::vector<const ST*> nodes;
    std::vector<FMCA::Matrix> bbvecActive;
    computeNodeAndBBActive(adaptiveTree, nodes, bbvecActive);
    saveBoxesToFile(bbvecActive, outputFileBoxes);

    return 0;
}

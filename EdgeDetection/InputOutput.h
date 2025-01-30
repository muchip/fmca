#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
//////////////////////////////////////////////////////////

#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"

using namespace FMCA;


// --------------------------------------------------------------------------------------------------
/**
 * @brief Save bounding boxes with slopes to file.
 *
 * This function saves the bounding boxes of the active leaves of a tree and
 * their corresponding slopes to two files in a format compatible with Python.
 * The boxes are saved as a list of squares: [(xmin, ymin), (xmax, ymin), (xmax,
 * ymax), (xmin, ymax)] and the slopes are saved as a list of scalars.
 *
 * @param slopes Map of active leaves to their slopes.
 * @param boxes_filename Name of the file to which the bounding boxes will be saved.
 * @param slopes_filename Name of the file to which the slopes will be saved.
 */
template <typename TreeType>
void saveBoxesWithSlopesToFile(const std::map<const TreeType*, Scalar>& slopes,
                               const std::string& boxes_filename,
                               const std::string& slopes_filename) {
  // Save bounding boxes
  std::ofstream boxes_outfile(boxes_filename);
  if (!boxes_outfile) {
    std::cerr << "Error opening boxes file: " << boxes_filename << std::endl;
    return;
  }
  // Save slopes
  std::ofstream slopes_outfile(slopes_filename);
  if (!slopes_outfile) {
    std::cerr << "Error opening slopes file: " << slopes_filename << std::endl;
    return;
  }
  boxes_outfile << "squares_with_boxes = [";
  slopes_outfile << "slopes = [";
  for (auto it = slopes.begin(); it != slopes.end(); ++it) {
    const auto* leaf = it->first;
    Scalar slope = it->second;
    auto xmin = leaf->bb()(0, 0);  // x min
    auto xmax = leaf->bb()(0, 1);  // x max
    auto ymin = leaf->bb()(1, 0);  // y min
    auto ymax = leaf->bb()(1, 1);  // y max
    // Boxes format: [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    boxes_outfile << "[(" << float(xmin) << ", " << float(ymin) << "), "
                  << "(" << float(xmax) << ", " << float(ymin) << "), "
                  << "(" << float(xmax) << ", " << float(ymax) << "), "
                  << "(" << float(xmin) << ", " << float(ymax) << ")]";
    // Slopes format: list of slopes
    slopes_outfile << slope;
    if (std::next(it) != slopes.end()) {
      boxes_outfile << ", ";
      slopes_outfile << ", ";
    }
  }
  boxes_outfile << "]" << std::endl;
  slopes_outfile << "]" << std::endl;
}



// --------------------------------------------------------------------------------------------------
/**
 * Saves the bounding boxes from the given collection to a file in a format
 * compatible with Python. Each box is represented as a list of coordinates
 * for its four corners: [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin,
 * ymax)].
 *
 * @tparam Bbvec Type of the bounding box collection.
 * @param bb Collection of bounding boxes to be saved.
 * @param filename Name of the file to which the bounding boxes will be saved.
 */
template <typename Bbvec>
void saveBoxesToFile(const Bbvec& bb, const std::string& filename) {
  std::ofstream outfile(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "squares = [";
  for (auto it = bb.begin(); it != bb.end(); ++it) {
    auto xmin = (*it)(0);  // x min
    auto xmax = (*it)(2);  // x max
    auto ymin = (*it)(1);  // y min
    auto ymax = (*it)(3);  // y max
    // Format for a square in Python
    outfile << "[(" << float(xmin) << ", " << float(ymin) << "), "
            << "(" << float(xmax) << ", " << float(ymin) << "), "
            << "(" << float(xmax) << ", " << float(ymax) << "), "
            << "(" << float(xmin) << ", " << float(ymax) << ")]";
    if (std::next(it) != bb.end()) {
      outfile << ", ";  // Add a comma between squares except for the last one
    }
  }
  outfile << "]" << std::endl;
  outfile.close();
  std::cout << "Squares saved to " << filename << std::endl;
}


// --------------------------------------------------------------------------------------------------
/**
 * Given a map of leaves to their slopes, this function generates data points
 * for a step function. The x-coordinates are the minimum x-coordinates of the
 * bounding boxes of the leaves, and the corresponding y-coordinate is the
 * slope of the leaf.
 *
 * The data is written to a file in the format:
 * x = [x1, x2, ...]
 * coeffs = [c1, c2, ...]
 * where x1, x2, ... are the x-coordinates and c1, c2, ... are the corresponding
 * slopes.
 *
 * @param slopes Map of leaves to their slopes.
 * @param outputFile Name of the file to which the data will be written.
 */

template <typename Derived>
void generateStepFunctionData(const std::map<const Derived*, Scalar>& slopes,
                              std::string outputFile) {
  std::vector<Scalar> x;       // Bounding box min x-coordinates
  std::vector<Scalar> coeffs;  // Slopes duplicated for step function

  for (const auto& [leaf, slope] : slopes) {
    const auto& bbox = leaf->bb();  // Assuming bb() gives the bounding box
    Scalar minX = bbox(0);          // Minimum x-coordinate
    Scalar maxX = bbox(1);          // Maximum x-coordinate

    // Append points for the step function
    x.push_back(minX);
    coeffs.push_back(slope);
  }

  // Write the data to a file
  std::ofstream file(outputFile);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << outputFile << std::endl;
    return;
  }

  file << std::fixed << std::setprecision(6);
  file << "x = [";
  for (size_t i = 0; i < x.size(); ++i) {
    file << x[i];
    if (i < x.size() - 1) file << ", ";
  }
  file << "]\n";

  file << "coeffs = [";
  for (size_t i = 0; i < coeffs.size(); ++i) {
    file << coeffs[i];
    if (i < coeffs.size() - 1) file << ", ";
  }
  file << "]\n";

  file.close();
  std::cout << "Data written to " << outputFile << std::endl;
}


// --------------------------------------------------------------------------------------------------
/**
 * Saves the contents of a vector to a file.
 *
 * This function opens a specified file and writes each element of the given
 * vector to the file, with each element on a new line. If the file cannot
 * be opened, it outputs an error message.
 *
 * @param vec The vector containing data to be saved.
 * @param filename The name of the file to which the vector data will be saved.
 */

void saveVectorToFile(const Vector& vec, const std::string& filename) {
  std::ofstream out_file(filename);
  if (out_file.is_open()) {
    for (size_t i = 0; i < vec.size(); ++i) {
      out_file << vec[i] << std::endl;
    }
    out_file.close();
    std::cout << "Vector saved to " << filename << std::endl;
  } else {
    std::cerr << "Error opening file " << filename << std::endl;
  }
}

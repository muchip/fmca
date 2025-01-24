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
 * Prints the given collection of intervals to the standard output in a format
 * compatible with Python. Each interval is represented as a tuple of two
 * coordinates: (start, end).
 *
 * @tparam Bbvec Type of the bounding box collection.
 * @param bb Collection of bounding boxes to be printed.
 */
template <typename Bbvec>
void printIntervalsPython(Bbvec& bb) {
  std::cout << "intervals = [";
  for (auto it = bb.begin(); it != bb.end(); ++it) {
    auto min = (*it)(0);  // Start of the interval
    auto max = (*it)(1);  // End of the interval
    std::cout << "(" << float(min) << ", " << float(max) << ")";
    if (std::next(it) != bb.end()) {
      std::cout
          << ", ";  // Add a comma between intervals except after the last one
    }
  }
  std::cout << "]" << std::endl;
}

// --------------------------------------------------------------------------------------------------
/**
 * Prints the bounding boxes from the given collection to the standard output
 * in a format compatible with Python. Each box is represented as a list of
 * coordinates for its four corners: [(xmin, ymin), (xmax, ymin), (xmax,
 * ymax), (xmin, ymax)].
 *
 * @tparam Bbvec Type of the bounding box collection.
 * @param bb Collection of bounding boxes to be printed.
 */
template <typename Bbvec>
void printBoxesPython2D(Bbvec& bb) {
  std::cout << "squares = [";
  for (auto it = bb.begin(); it != bb.end(); ++it) {
    auto xmin = (*it)(0);  // x min
    auto xmax = (*it)(2);  // x max
    auto ymin = (*it)(1);  // y min
    auto ymax = (*it)(3);  // y max

    // Format for a square in Python: [(xmin, ymin), (xmax, ymin), (xmax,
    // ymax), (xmin, ymax)]
    std::cout << "[(" << float(xmin) << ", " << float(ymin) << "), "
              << "(" << float(xmax) << ", " << float(ymin) << "), "
              << "(" << float(xmax) << ", " << float(ymax) << "), "
              << "(" << float(xmin) << ", " << float(ymax) << ")]";

    if (std::next(it) != bb.end()) {
      std::cout << ", ";  // Add a comma between squares except for the last one
    }
  }
  std::cout << "]" << std::endl;
}

// --------------------------------------------------------------------------------------------------

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
 * Computes a vector of coefficients for each point in the samplet tree where
 * each coefficient is the slope of the corresponding leaf.
 *
 * @param slopes A map of leaves to their slopes
 * @param tdata The data in the samplet basis
 * @param Pointsslopes_filename The name of the file to which the coefficients
 * will be saved
 *
 * @return A vector of coefficients, where each coefficient is the slope of the
 * corresponding leaf
 */
template <typename TreeType>
Vector PointsWithSlopes(const std::map<const TreeType*, Scalar>& slopes,
                        const Vector& tdata) {

    Vector PointsCoeffs = Eigen::VectorXd::Zero(tdata.size());

    // Iterate over each leaf -> slope mapping
    for (const auto& [leaf, slope] : slopes) {
        int start = leaf->start_index();
        int count = leaf->nsamplets();
        // Fill the corresponding segment in PointsCoeffs with 'slope'
        PointsCoeffs.segment(start, count) = 
            Eigen::VectorXd::Ones(count) * slope;
    }
    return PointsCoeffs;
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
 * Generates data for a step function based on the slopes of the leaves of a
 * SampletTree.
 *
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

// --------------------------------------------------------------------------------------------------
void generatePlottingScript1D(const std::string& outputFile,
                              const std::string& path_step_txt_file,
                              const Matrix& Points, const Vector& f) {
  std::ofstream file(outputFile);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << outputFile << std::endl;
    return;
  }

  // Write the Python script header
  file << "import matplotlib.pyplot as plt\n";
  file << "import numpy as np\n\n";

  // Write x_f values array
  file << "x_f = np.array([";
  for (Index i = 0; i < Points.cols(); ++i) {
    file << Points(i);
    if (i < Points.cols() - 1) file << ", ";
  }
  file << "])\n\n";

  // Write the function values array
  file << "f = np.array([";
  for (Index i = 0; i < f.size(); ++i) {
    file << f(i);
    if (i < f.size() - 1) file << ", ";
  }
  file << "])\n\n";

  // Write the rest of the script using raw string literal
  file << R"(# Load data from the file
file_path = ")"
       << path_step_txt_file << R"("
with open(file_path, "r") as file:
    lines = file.readlines()

# Define a replacement for `nan` as `np.nan`
x = eval(lines[0].split('=')[1].strip())
coeffs = lines[1].split('=')[1].strip()

# Replace 'nan' with 'np.nan' to handle them correctly
coeffs = coeffs.replace('nan', 'np.nan')
coeffs = eval(coeffs)

# Filter out nan values
valid_indices = ~np.isnan(coeffs)
x = np.array(x)[valid_indices]
coeffs = np.array(coeffs)[valid_indices]

# Plot the step function
fig, ax = plt.subplots(figsize=(9, 6))  # Increase figure size
plt.step(x, coeffs, where="post", label="Slope coeff decay", linewidth=3, color='blue')
plt.plot(x_f, f, linewidth=3, color='orange', label="$f(x)$")

# Add horizontal lines
horizontal_lines = [-0.5, -1, -1.5, -2]
colors = ['red', 'lime', 'magenta', 'cyan']
for i in range(len(horizontal_lines)):
    ax.hlines(
        horizontal_lines[i], 
        -1.1, 
        1.1, 
        color=colors[i], 
        linestyles='dashed', 
        label=f"Slope = {horizontal_lines[i]}", 
        linewidth=2
    )

# Improve labels, ticks, and title
plt.ylabel("Slope", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)

# Improve legend
plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.7, 0.45), fontsize=14)

# Tight layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
)";

  file.close();
  std::cout << "Python plotting script generated at " << outputFile
            << std::endl;
}
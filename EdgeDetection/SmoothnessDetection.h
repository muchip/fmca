#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
//////////////////////////////////////////////////////////
#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeInitializer_RandomProjectionTree.h"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/Clustering/RandomTree1D.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Plotter3D.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "../TestPDE/read_files_txt.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the maximum level of a tree.
 *
 * Iterate through the nodes of a tree and return the maximum level of the tree.
 *
 * @param st tree.
 * @return maximum level of the tree.
 */
template <typename Derived>
Index computeMaxLevel(const SampletTreeBase<Derived>& st) {
  Index max_level = 0;
  for (const auto& node : st) {
    max_level = std::max(max_level, node.level());
  }
  return max_level;
}
// template <typename Node>
// Index computeMaxLevel(const Node& node) {
//   Index max_level = node.level();  // Get the current node's level
//   for (Index i = 0; i < node.nSons(); ++i) {
//     max_level = std::max(max_level, computeMaxLevel(node.sons(i)));
//   }

//   return max_level;
// }

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Compute the maximum coefficient for a node.
 *
 * Given a node, computes the maximum of the absolute value of the
 * coefficients in the samplet basis.
 *
 * @param[in] node node of the tree.
 * @param[in] tdata data in samplet basis.
 * @return maximum coefficient of the node.
 */
template <typename Derived>
Scalar computeMaxCoeffNode(const Derived& node, const Vector& tdata) {
  int level = node.level();
  auto segment = tdata.segment(node.start_index(), node.nsamplets());
  Scalar coefficient = segment.cwiseAbs().maxCoeff();
  return coefficient;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * For each level of the tree, printMaxCoefficientsPerLevel gives the largest
 * coefficient.
 *
 * This function traverses a samplet tree and calculates the maximum of
 * coefficients for each level using the provided data in samplet basis. It
 * prints the maximum coefficients along with the corresponding levels.
 *
 * @tparam Derived The derived type of the samplet tree base.
 * @param st The samplet tree for which maximum coefficients per level are
 * computed.
 * @param tdata The data in the samplet basis used to compute the coefficients.
 */
template <typename Derived>
void printMaxCoefficientsPerLevel(const SampletTreeBase<Derived>& st,
                                  const Vector& tdata) {
  // Map to store the Scalar maximum coefficient for each int level
  std::map<int, Scalar> levelMax;
  Vector levels;
  Vector values;

  for (const auto& node : st) {
    if (node.nsamplets() <= 0) {
      std::cerr << "Warning: Node with no samplets at level " << node.level()
                << std::endl;
      continue;
    }
    int level = node.level();
    auto segment = tdata.segment(node.start_index(), node.nsamplets());
    // We consider the segment of the samplet transformed tdata that correspond
    // to the node cluster (we are in practice considering the points that
    // belong to the cluster and we take the max coefficient of the absolute
    // value --> worst case)
    Scalar coefficient =
        segment.cwiseAbs()
            .maxCoeff();  // alternative: Scalar coefficient = segment.norm();

    if (levelMax.find(level) == levelMax.end()) {
      // if we did not consider that level before, the simpy add the coeffcient
      levelMax[level] = coefficient;
    } else {
      // if we have already encounter this level in previous iteration, save the
      // coefficient just if it is bigger than the previous one
      levelMax[level] = std::max(levelMax[level], coefficient);
    }
  }

  // print the results (the second and third "for loops" are just useful to
  // paste and copy the results and visualize them in Python)
  levels.resize(levelMax.size());
  values.resize(levelMax.size());
  int index = 0;
  for (const auto& [level, maxCoefficient] : levelMax) {
    std::cout << "Level " << level << ": Max Coefficient = " << maxCoefficient
              << std::endl;
    levels(index) = level;
    values(index) = maxCoefficient;
    ++index;
  }

  std::cout << "Levels = [";
  for (int i = 0; i < levels.size(); ++i) {
    std::cout << levels(i);
    if (i < levels.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "values = [";
  for (int i = 0; i < values.size(); ++i) {
    std::cout << values(i);
    if (i < values.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
template <typename Derived>
void traverseAndStackCoefficients(
    const SampletTreeBase<Derived>& tree, const Vector& tdata,
    std::map<const Derived*, std::vector<Scalar>>& leafCoefficients) {
  // A stack to simulate Depth-First Search (DFS) traversal, with current node
  // and level.
  std::vector<std::pair<const Derived*, size_t>> nodeStack;

  // Vector to keep track of coefficients for nodes as we traverse the tree.
  std::vector<Scalar> coefficientsStack;

  // Push the root node onto the stack with level 0.
  nodeStack.emplace_back(&tree.derived(), 0);

  // Main loop for DFS traversal.
  while (!nodeStack.empty()) {
    // Get the current node and its level.
    auto [currentNode, currentLevel] = nodeStack.back();
    nodeStack.pop_back();

    // Ensure the coefficients stack matches the current path.
    if (coefficientsStack.size() > currentLevel) {
      coefficientsStack.resize(currentLevel);
    }

    // Compute the coefficient for the current node based on `tdata`.
    Scalar coeff = computeMaxCoeffNode(*currentNode, tdata);

    // Push the computed coefficient onto the coefficients stack.
    coefficientsStack.push_back(coeff);

    // Check if the current node is a leaf node.
    if (currentNode->nSons() == 0) {
      // Store the current stack of coefficients for this leaf node.
      leafCoefficients[currentNode] = coefficientsStack;
    } else {
      // Push children nodes onto the stack with their respective levels.
      for (int i = currentNode->nSons() - 1; i >= 0; --i) {
        nodeStack.emplace_back(std::addressof(currentNode->sons(i)),
                               currentLevel + 1);
      }
    }
  }
}

template <typename TreeType>
std::map<const TreeType*, Scalar> computeLinearRegressionSlope(
    const std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients, const Scalar& dtilde) {
  std::map<const TreeType*, Scalar> slopes;

  for (const auto& [leaf, coefficients] : leafCoefficients) {
    size_t n = coefficients.size();

    // Exclude the first and last coefficients
    std::vector<Scalar> x, y;
    int counter_small_coefficients = 0;
    // Scalar log10_sum_small_coefficients = 0.0;
    for (size_t i = 1; i < n; ++i) {
      x.push_back(static_cast<Scalar>(i));      // Levels 1, 2, ..., n-2
      y.push_back(std::log2(coefficients[i]));  // Log2 of the coefficients
      if (coefficients[i] < 1e-6) {
        counter_small_coefficients++;
        // log10_sum_small_coefficients += std::log10(coefficients[i]);
      }
    }
    // if counter is greather that half of the coefficients, set the slope to -10
    if (counter_small_coefficients > n / 3) {
      //  Scalar avg_log10_small_coefficients = log10_sum_small_coefficients / counter_small_coefficients;
      //  int slope = static_cast<int>(std::round(avg_log10_small_coefficients));
      slopes[leaf] = -dtilde;
    } else {
      // Compute the slope using linear regression
      Scalar sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
      size_t m = x.size();
      for (size_t i = 0; i < m; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
      }

      Scalar slope = (m * sumXY - sumX * sumY) / (m * sumX2 - sumX * sumX);
      slopes[leaf] = slope;
    }
  }

  return slopes;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/**
 * For each level of the tree, computes and prints the average coefficient.
 *
 * This function traverses a samplet tree and calculates the average of
 * coefficients for each level using the provided data in samplet basis.
 * It prints the average coefficients along with the corresponding levels.
 *
 * @tparam Derived The derived type of the samplet tree base.
 * @param st The samplet tree for which average coefficients per level are
 * computed.
 * @param tdata The data in the samplet basis used to compute the
 * coefficients.
 */
template <typename Derived>
void printAverageCoefficientsPerLevel(const SampletTreeBase<Derived>& st,
                                      const Vector& tdata) {
  std::map<int, std::pair<Scalar, int>> levelSumAndCount;
  Vector levels;
  Vector averages;

  for (const auto& node : st) {
    int level = node.level();
    Scalar coefficient =
        tdata.segment(node.start_index(), node.nsamplets()).norm();

    if (levelSumAndCount.find(level) == levelSumAndCount.end()) {
      levelSumAndCount[level] = {coefficient, 1};
    } else {
      levelSumAndCount[level].first += coefficient;  // Add to sum
      levelSumAndCount[level].second += 1;           // Increment count
    }
  }

  levels.resize(levelSumAndCount.size());
  averages.resize(levelSumAndCount.size());
  int index = 0;
  for (const auto& [level, sumAndCount] : levelSumAndCount) {
    Scalar average = sumAndCount.first / sumAndCount.second;
    std::cout << "Level " << level << ": Average Coefficient = " << average
              << std::endl;
    levels(index) = level;
    averages(index) = average;
    ++index;
  }

  std::cout << "Levels = [";
  for (int i = 0; i < levels.size(); ++i) {
    std::cout << levels(i);
    if (i < levels.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "Averages = [";
  for (int i = 0; i < averages.size(); ++i) {
    std::cout << averages(i);
    if (i < averages.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/**
 * Computes an adaptive tree from a given samplet tree and data.
 *
 * This function performs an adaptive tree search on the provided samplet tree
 * based on the input data and a threshold for active leaves. The adaptive
 * tree is constructed by keeping nodes that have significant coefficients
 * with respect to the threshold.
 *
 * @tparam Derived The derived type of the samplet tree base.
 * @param st The samplet tree on which the adaptive tree is computed.
 * @param tdata The data in the samplet basis used to compute the adaptive
 * tree.
 * @param threshold_active_leaves The threshold used to determine active
 * leaves based on the squared norm of the data.
 * @return A vector of pointers to the nodes of the adaptive tree.
 */
template <typename ClusterTreeType>
std::vector<const H2SampletTree<ClusterTreeType>*> computeAdaptiveTree(
    const H2SampletTree<ClusterTreeType>& st, const Vector& tdata,
    const Scalar& threshold_active_leaves) {
  std::vector<const H2SampletTree<ClusterTreeType>*> adaptive_tree =
      adaptiveTreeSearch(st, tdata,
                         threshold_active_leaves * tdata.squaredNorm());
  const FMCA::Index nclusters = std::distance(st.begin(), st.end());

  FMCA::Vector thres_tdata = tdata;
  thres_tdata.setZero();
  FMCA::Index nnz = 0;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const H2SampletTree<ClusterTreeType>& node = *(adaptive_tree[i]);
      const FMCA::Index ndist =
          node.is_root() ? node.Q().cols() : node.nsamplets();
      thres_tdata.segment(node.start_index(), ndist) =
          tdata.segment(node.start_index(), ndist);
      nnz += ndist;
    }
  }
  std::cout << "active coefficients: " << nnz << " / " << tdata.rows()
            << std::endl;
  std::cout << "tree error: " << (thres_tdata - tdata).norm() / tdata.norm()
            << std::endl;

  return adaptive_tree;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief computeBBActive computes the bounding boxes of the active leaves of
 * an adaptive tree.
 *
 * @param[in] adaptive_tree vector of pointers to nodes of the adaptive tree
 * @return a vector of bounding boxes for the active leaves of the tree
 */
template <typename ClusterTreeType>
std::vector<FMCA::Matrix> computeBBActive(
    const std::vector<const H2SampletTree<ClusterTreeType>*>& adaptive_tree) {
  std::vector<FMCA::Matrix> bbvec_active;
  for (const auto* it : adaptive_tree) {
    if (it != nullptr && !it->nSons()) {
      const auto& node = *it;  // reference to the dereferenced pointer
      bbvec_active.push_back(node.bb());
    }
  }
  return bbvec_active;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
template <typename ClusterTreeType>
void computeNodeAndBBActive(
    const std::vector<const H2SampletTree<ClusterTreeType>*>& adaptive_tree,
    std::vector<const H2SampletTree<ClusterTreeType>*>& nodes,
    std::vector<FMCA::Matrix>& bbvec_active) {
  nodes.clear();
  bbvec_active.clear();
  for (const auto* it : adaptive_tree) {
    if (it != nullptr && !it->nSons()) {
      const auto& node = *it;  // Reference to the dereferenced pointer
      nodes.push_back(it);
      bbvec_active.push_back(node.bb());
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
/**
 * For each level of the tree, printMaxCoefficientsPerBranch gives the largest
 * coefficient on the branch from the leaf node to the root.
 *
 * @param leafNode the leaf node of the branch
 * @param tdata data in samplet basis
 * @return largest coefficient (worst case) for each tree level on the branch.
 */
template <typename ClusterTreeType>
void printMaxCoefficientsPerBranch(
    const H2SampletTree<ClusterTreeType>* leafNode, const Vector& tdata) {
  // Map to store the Scalar maximum coefficient for each int level
  std::map<int, Scalar> levelMax;
  Vector levels;
  Vector values;

  // Traverse the branch from the leaf node to the root
  const H2SampletTree<ClusterTreeType>* currentNode = leafNode;
  while (currentNode != nullptr) {
    int level = currentNode->level();
    auto segment =
        tdata.segment(currentNode->start_index(), currentNode->nsamplets());
    Scalar coefficient = segment.cwiseAbs().maxCoeff();

    if (levelMax.find(level) == levelMax.end()) {
      levelMax[level] = coefficient;
    } else {
      levelMax[level] = std::max(levelMax[level], coefficient);
    }

    // Move to the parent node
    currentNode = currentNode->is_root() ? nullptr : &currentNode->dad();
  }

  // Print the results
  levels.resize(levelMax.size());
  values.resize(levelMax.size());
  int index = 0;
  for (const auto& [level, maxCoefficient] : levelMax) {
    // std::cout << "Level " << level << ": Max Coefficient = " <<
    // maxCoefficient
    //           << std::endl;
    levels(index) = level;
    values(index) = maxCoefficient;
    ++index;
  }

  std::cout << "Levels = [";
  for (int i = 0; i < levels.size(); ++i) {
    std::cout << levels(i);
    if (i < levels.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "values = [";
  for (int i = 0; i < values.size(); ++i) {
    std::cout << values(i);
    if (i < values.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
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

template <typename ClusterTreeType>
void generateStepFunctionData(
    const std::map<const H2SampletTree<ClusterTreeType>*, Scalar>& slopes,
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
    x.push_back(maxX);
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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
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
            
    // Set the max coefficient for the node
    node.derived().setMaxCoeff(coefficient);

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
 * @param tdata The data in the samplet basis used to compute the coefficients.
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
 * based on the input data and a threshold for active leaves. The adaptive tree
 * is constructed by keeping nodes that have significant coefficients with
 * respect to the threshold.
 *
 * @tparam Derived The derived type of the samplet tree base.
 * @param st The samplet tree on which the adaptive tree is computed.
 * @param tdata The data in the samplet basis used to compute the adaptive tree.
 * @param threshold_active_leaves The threshold used to determine active leaves
 *        based on the squared norm of the data.
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
 * @brief computeBBActive computes the bounding boxes of the active leaves of an
 * adaptive tree.
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
 * coordinates for its four corners: [(xmin, ymin), (xmax, ymin), (xmax, ymax),
 * (xmin, ymax)].
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

    // Format for a square in Python: [(xmin, ymin), (xmax, ymin), (xmax, ymax),
    // (xmin, ymax)]
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

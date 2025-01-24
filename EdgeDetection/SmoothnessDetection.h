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
#include <map>
#include <utility>
//////////////////////////////////////////////////////////

#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "read_files_txt.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
// FMCA::NystromSampletMoments<SampletInterpolator>;
// FMCA::MinNystromSampletMoments<SampletInterpolator>

using namespace FMCA;

// --------------------------------------------------------------------------------------------------
/**
 * @brief Compute the maximum level of a tree.
 *
 * Iterate through the nodes of a tree and return the maximum level of the tree.
 *
 * @param st tree.
 * @return maximum level of the tree.
 */
template <typename TreeType>
Index computeMaxLevel(const TreeType& st) {
  Index max_level = 0;
  for (const auto& node : st) {
    max_level = std::max(max_level, node.level());
  }
  return max_level;
}

// --------------------------------------------------------------------------------------------------
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
std::optional<Scalar> computeMaxCoeffNode(const Derived& node,
                                          const Vector& tdata) {
  if (node.nsamplets() <= 0 || node.start_index() < 0 ||
      node.start_index() + node.nsamplets() > tdata.size()) {
    return std::nullopt;  // Return empty optional for invalid nodes
  }
  auto segment = tdata.segment(node.start_index(), node.nsamplets());
  Scalar coefficient = segment.cwiseAbs().maxCoeff();
  return coefficient;
}

// --------------------------------------------------------------------------------------------------
/**
 * For each level of the tree, printMaxCoefficientsPerLevel gives the largest
 * coefficient.
 *
 * This function traverses a samplet tree and calculates the maximum of
 * coefficients for each level using the provided data in samplet basis. It
 * prints the maximum coefficients along with the corresponding levels.
 *
 * @tparam TreeType The derived type of the samplet tree base.
 * @param st The samplet tree for which maximum coefficients per level are
 * computed.
 * @param tdata The data in the samplet basis used to compute the coefficients.
 */
template <typename TreeType>
void printMaxCoefficientsPerLevel(const TreeType& st,
                                  const Vector& tdata) {
  // Map to store the Scalar maximum coefficient for each int level
  std::map<int, Scalar> levelMax;
  Vector levels;
  Vector values;

  for (const auto& node : st) {
    if (node.nsamplets() <= 0) {
      // std::cerr << "Warning: Node with no samplets at level " << node.level()
      //           << std::endl;
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



// --------------------------------------------------------------------------------------------------
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
template <typename TreeType>
void printAverageCoefficientsPerLevel(const TreeType& st,
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


// --------------------------------------------------------------------------------------------------
template <typename TreeType>
void traverseAndStackCoefficients(
    const TreeType& tree, 
    const Vector& tdata,
    std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients) 
{
    // Stack used for DFS (node + level in the tree).
    std::vector<std::pair<const TreeType*, size_t>> nodeStack{{&tree.derived(), 0}};
    // Accumulates coefficients along the path to the current node.
    std::vector<Scalar> coefficientsStack;

    while (!nodeStack.empty()) {
        auto [currentNode, currentLevel] = nodeStack.back();
        nodeStack.pop_back();

        // If we climbed back up the tree, shrink the coefficients stack.
        if (coefficientsStack.size() > currentLevel) {
            coefficientsStack.resize(currentLevel);
        }

        // Compute the coefficient for the current node.
        std::optional<Scalar> coeffOpt = computeMaxCoeffNode(*currentNode, tdata);
        if (!coeffOpt) {
            // If no valid coefficient, skip this node entirely.
            continue;
        }
        // Add the coefficient to the current path stack.
        coefficientsStack.push_back(*coeffOpt);

        // Check if this node is effectively a leaf.
        bool noSamplets = (currentNode->nsamplets() <= 0);
        bool isLeaf     = (currentNode->nSons() == 0);

        if (noSamplets || isLeaf) {
            // Treat it as a leaf: store coefficients for this node
            // and do not visit children.
            leafCoefficients[currentNode] = coefficientsStack;
        } else {
            // Otherwise, push children onto the stack (reverse order for DFS).
            for (int i = currentNode->nSons() - 1; i >= 0; --i) {
                nodeStack.emplace_back(&currentNode->sons(i), currentLevel + 1);
            }
        }
    }
}




// --------------------------------------------------------------------------------------------------
/**
 * Computes an adaptive tree from a given samplet tree and data.
 *
 * This function performs an adaptive tree search on the provided samplet tree
 * based on the input data and a threshold for active leaves. The adaptive
 * tree is constructed by keeping nodes that have significant coefficients
 * with respect to the threshold.
 *
 * @tparam TreeType The derived type of the samplet tree base.
 * @param st The samplet tree on which the adaptive tree is computed.
 * @param tdata The data in the samplet basis used to compute the adaptive
 * tree.
 * @param threshold_active_leaves The threshold used to determine active
 * leaves based on the squared norm of the data.
 * @return A vector of pointers to the nodes of the adaptive tree.
 */
template <typename TreeType>
std::vector<const TreeType*> computeAdaptiveTree(
    const TreeType& st, const Vector& tdata,
    const Scalar& threshold_active_leaves) {
  std::vector<const TreeType*> adaptive_tree = adaptiveTreeSearch(
      st, tdata, threshold_active_leaves * tdata.squaredNorm());
  const FMCA::Index nclusters = std::distance(st.begin(), st.end());

  FMCA::Vector thres_tdata = tdata;
  thres_tdata.setZero();
  FMCA::Index nnz = 0;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const TreeType& node = *(adaptive_tree[i]);
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



// --------------------------------------------------------------------------------------------------
/**
 * @brief computeBBActive computes the bounding boxes of the active leaves of
 * an adaptive tree.
 *
 * @param[in] adaptive_tree vector of pointers to nodes of the adaptive tree
 * @return a vector of bounding boxes for the active leaves of the tree
 */
template <typename TreeType>
std::vector<FMCA::Matrix> computeBBActive(
    const std::vector<const TreeType*>& adaptive_tree) {
  std::vector<FMCA::Matrix> bbvec_active;
  for (const auto* it : adaptive_tree) {
    if (it != nullptr && !it->nSons()) {
      const auto& node = *it;  // reference to the dereferenced pointer
      bbvec_active.push_back(node.bb());
    }
  }
  return bbvec_active;
}



// --------------------------------------------------------------------------------------------------
template <typename TreeType>
void computeNodeAndBBActive(
    const std::vector<const TreeType*>& adaptive_tree,
    std::vector<const TreeType*>& nodes,
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



// --------------------------------------------------------------------------------------------------
/**
 * For each level of the tree, printMaxCoefficientsPerBranch gives the largest
 * coefficient on the branch from the leaf node to the root.
 *
 * @param leafNode the leaf node of the branch
 * @param tdata data in samplet basis
 * @return largest coefficient (worst case) for each tree level on the branch.
 */
template <typename TreeType>
void printMaxCoefficientsPerBranch(const TreeType* leafNode,
                                   const Vector& tdata) {
  // Map to store the Scalar maximum coefficient for each int level
  std::map<int, Scalar> levelMax;
  Vector levels;
  Vector values;

  // Traverse the branch from the leaf node to the root
  const TreeType* currentNode = leafNode;
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

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
/**
 * @brief traverseAndStackCoefficientsAndDiameters traverses the samplet tree, computing
 * the maximum coefficient and diameter for each node, and storing the
 * results in a map.
 *
 * This function performs a depth-first traversal of the samplet tree, computing
 * the maximum coefficient and diameter for each node using the provided data
 * in the samplet basis. For each leaf node, it stores the coefficients and
 * diameters of the path from the root to the leaf in the provided map.
 *
 * @param tree The samplet tree to traverse.
 * @param tdata The data in the samplet basis used to compute the coefficients.
 * @param leafData A map of leaf nodes to pairs of vectors containing the
 * coefficients and diameters of the path from the root to the leaf.
 */
template <typename TreeType>
void traverseAndStackCoefficientsAndDiameters(
    const TreeType& tree, 
    const Vector&    tdata,
    std::map<const TreeType*, std::pair<std::vector<Scalar>, std::vector<Scalar>>>& leafData)
{
    using StackItem = std::tuple<const TreeType*, size_t, const TreeType*>;
    std::vector<StackItem> nodeStack;
    nodeStack.emplace_back(&tree.derived(), 0, static_cast<const TreeType*>(nullptr));

    // Path stacks
    std::vector<Scalar> coefficientsStack;
    std::vector<Scalar> diametersStack;

    while (!nodeStack.empty()) {
        auto [currentNode, currentLevel, parentNode] = nodeStack.back();
        nodeStack.pop_back();

        // If we are returning from a deeper level, shrink stacks accordingly
        if (coefficientsStack.size() > currentLevel) {
            coefficientsStack.resize(currentLevel);
            diametersStack.resize(currentLevel);
        }

        // Compute the coefficient for this node (use "computeMaxCoeffNode")
        std::optional<Scalar> coeffOpt = computeMaxCoeffNode(*currentNode, tdata);
        if (!coeffOpt) {
            // Skip children if no valid coefficient
            continue;
        }

        // Compute the diameter as the norm of the last column of bb()
        Scalar diam = (currentNode->bb().col(currentNode->bb().rows())).norm();

        coefficientsStack.push_back(*coeffOpt);
        diametersStack.push_back(diam);

        bool noSamplets = (currentNode->nsamplets() <= 0);
        bool isLeaf     = (currentNode->nSons() == 0);

        if (isLeaf) {
            // A real leaf => store the entire path
            leafData[currentNode] = std::make_pair(coefficientsStack, diametersStack);
        }
        else if (noSamplets) {
            // Node has "no samplets". Treat the *parent* as a leaf
            if (parentNode) {
                // Pop the current node's coefficient/diam
                auto parentCoeffs = coefficientsStack;
                parentCoeffs.pop_back();
                auto parentDiams = diametersStack;
                parentDiams.pop_back();

                leafData[parentNode] = std::make_pair(std::move(parentCoeffs), std::move(parentDiams));
            }
            // Do NOT continue to children
        }
        else {
            // Normal internal node => push children for DFS
            for (int i = currentNode->nSons() - 1; i >= 0; --i) {
                nodeStack.emplace_back(&currentNode->sons(i), currentLevel + 1, currentNode);
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

// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H
#define FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H

namespace FMCA {

/**
 * @brief Analyzes samplet coefficients across tree levels.
 *
 * @tparam TreeType Type of the tree structure being analyzed.
 *
 * This class traverses a tree structure and analyzes samplet coefficients,
 * computing sum of squared coefficients per level and per node. It supports
 * depth-first traversal with coefficient and diameter tracking for each path
 * from root to leaf.
 */

template <typename TreeType>
class SampletCoefficientsAnalyzer {
 public:
  using LeafDataMap =
      std::map<const TreeType*,
               std::pair<std::vector<Scalar>, std::vector<Scalar>>>;
  using LevelMap = std::map<int, Scalar>;

  SampletCoefficientsAnalyzer() = default;

  SampletCoefficientsAnalyzer(const TreeType& tree, const Vector& tdata) {
    init(tree, tdata);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  void init(const TreeType& tree, const Vector& tdata) {
    tree_ptr_ = &tree;
    tdata_ptr_ = &tdata;
    max_level_ = computeMaxLevel(tree);
    sum_squared_per_level_ = getSumSquaredPerLevel(tree, tdata);
  }

  //////////////////////////////////////////////////////////////////////////////
  Index computeMaxLevel(const TreeType& tree) {
    Index max_level = 0;
    for (const auto& node : tree) {
      max_level = std::max(max_level, node.level());
    }
    return max_level;
  }

  //////////////////////////////////////////////////////////////////////////////
  Scalar computeSumSquaredCoeffNode(const TreeType& node, const Vector& tdata) {
    if (node.nsamplets() <= 0 || node.start_index() < 0 ||
        node.start_index() + node.nsamplets() > tdata.size() ||
        node.block_size() <= 0) {
      return std::numeric_limits<Scalar>::quiet_NaN();  // Return NaN
    }
    auto segment = tdata.segment(node.start_index(), node.nsamplets());
    Scalar coefficient = segment.cwiseProduct(segment).sum();
    return coefficient;
  }

  //////////////////////////////////////////////////////////////////////////////
  LevelMap getSumSquaredPerLevel(const TreeType& tree, const Vector& tdata) {
    LevelMap levelCoeff;
    for (const auto& node : tree) {
      auto level = node.level();
      auto coefficient = computeSumSquaredCoeffNode(node, tdata);
      // Skip nodes that returned NaN (invalid nodes)
      if (std::isnan(coefficient)) {
        continue;
      }
      if (levelCoeff.find(level) == levelCoeff.end()) {
        levelCoeff[level] = coefficient;
      } else {
        levelCoeff[level] += coefficient;
      }
    }
    return levelCoeff;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
 * @brief Traverses tree and collects coefficient and diameter paths to leaves.
 *
 * Performs depth-first traversal of the tree, maintaining stacks of
 * normalized coefficients and diameters along each path from root to leaf.
 * For each leaf node, stores the complete path in the output map.
 *
 * Normalization: Coefficients are normalized by taking the square root of
 * the sum of squared coefficients. 
 *
 * @param tree The tree structure to traverse.
 * @param tdata Vector of samplet coefficient data.
 * @param leaf_data Output map storing (coefficients, diameters) paths for
 *                  each leaf node.
 */
  void traverseAndStackCoefficientsAndDiametersL2Norm(const TreeType& tree,
                                                      const Vector& tdata,
                                                      LeafDataMap& leafData) {
    Scalar N = tdata.rows();
    using StackItem = std::tuple<const TreeType*, size_t, const TreeType*>;
    std::vector<StackItem> nodeStack;
    // compute sum of squared coefficients per level
    LevelMap levelNorms = getSumSquaredPerLevel(tree, tdata);
    // Convert to sqrt for normalization
    for (auto& [level, sumSquared] : levelNorms) {
      levelNorms[level] = std::sqrt(sumSquared);
    }
    // Start from the root
    nodeStack.emplace_back(&tree.derived(), 0, nullptr);
    std::vector<Scalar> coefficientsStack;
    std::vector<Scalar> diametersStack;
    while (!nodeStack.empty()) {
      auto [currentNode, currentLevel, parentNode] = nodeStack.back();
      nodeStack.pop_back();
      // If we backtrack to a shallower level, resize the stacks
      if (coefficientsStack.size() > currentLevel) {
        coefficientsStack.resize(currentLevel);
        diametersStack.resize(currentLevel);
      }
      // Compute sum of squares for this node
      Scalar coeff = computeSumSquaredCoeffNode(*currentNode, tdata);
      Scalar diam = (currentNode->bb().col(2)).norm();
      // Normalize coefficient by level norm
      Scalar normalizedCoeff = std::sqrt(coeff);
      if (levelNorms.find(currentLevel) != levelNorms.end() &&
          levelNorms[currentLevel] > 0) {
        normalizedCoeff = normalizedCoeff; // / ( std::sqrt(currentNode->block_size()) ); // /= levelNorms[currentLevel];
      }
      coefficientsStack.push_back(normalizedCoeff);
      diametersStack.push_back(diam);
      bool isLeaf = (currentNode->nSons() == 0);
      if (isLeaf) {
        leafData[currentNode] =
            std::make_pair(coefficientsStack, diametersStack);
      } else {
        // Push children (in reverse order to process left child first)
        for (int i = currentNode->nSons() - 1; i >= 0; --i) {
          nodeStack.emplace_back(&currentNode->sons(i), currentLevel + 1,
                                 currentNode);
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
 private:
  const TreeType* tree_ptr_ = nullptr;
  const Vector* tdata_ptr_ = nullptr;
  Index max_level_ = 0;
  LevelMap sum_squared_per_level_;
};

}  // namespace FMCA

#endif  // FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H

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
    max_coeff_per_level_ = getMaxCoefficientsPerLevel(tree, tdata);
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
  Scalar computeMaxCoeffNode(const TreeType& node, const Vector& tdata) {
    if (node.nsamplets() <= 0 || node.start_index() < 0 ||
        node.start_index() + node.nsamplets() > tdata.size() ||
        node.block_size() <= 0) {
      return std::numeric_limits<Scalar>::quiet_NaN();  // Return NaN
    }
    auto segment = tdata.segment(node.start_index(), node.nsamplets());
    Scalar coefficient = segment.cwiseAbs().maxCoeff();
    return coefficient;
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
  LevelMap getMaxCoefficientsPerLevel(const TreeType& tree,
                                      const Vector& tdata) {
    LevelMap levelCoeff;
    for (const auto& node : tree) {
      auto level = node.level();
      auto coefficient = computeMaxCoeffNode(node, tdata);

      // Skip nodes that returned NaN (invalid nodes)
      if (std::isnan(coefficient)) {
        continue;
      }

      if (levelCoeff.find(level) == levelCoeff.end()) {
        levelCoeff[level] = coefficient;
      } else {
        levelCoeff[level] = std::max(levelCoeff[level], coefficient);
      }
    }
    return levelCoeff;
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
  void traverseAndStackCoefficientsAndDiameters(const TreeType& tree,
                                                const Vector& tdata,
                                                LeafDataMap& leafData) {
    using StackItem = std::tuple<const TreeType*, size_t,
                                 const TreeType*>;  // (node, level, parent)
    std::vector<StackItem> nodeStack;
    nodeStack.emplace_back(&tree.derived(), 0,
                           static_cast<const TreeType*>(nullptr));

    Scalar N = tdata.rows();

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

      // Compute the coefficient for this node - now returns NaN for invalid
      // nodes
      Scalar coeff = computeMaxCoeffNode(*currentNode, tdata);

      // Compute the diameter as the norm of the last column of bb()
      Scalar diam =
          (currentNode->bb().col(2)).norm();  // currentNode->block_size() / N;

      // Add current values to stacks
      coefficientsStack.push_back(coeff);
      diametersStack.push_back(diam);

      bool isLeaf = (currentNode->nSons() == 0);
      if (isLeaf) {
        // A real leaf => store the entire path including possibly NaN values
        leafData[currentNode] =
            std::make_pair(coefficientsStack, diametersStack);
      } else {
        // Normal internal node => push children for DFS
        for (int i = currentNode->nSons() - 1; i >= 0; --i) {
          nodeStack.emplace_back(&currentNode->sons(i), currentLevel + 1,
                                 currentNode);
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  void traverseAndStackCoefficientsAndDiametersL2Norm(const TreeType& tree,
                                                      const Vector& tdata,
                                                      LeafDataMap& leafData) {
    Scalar N = tdata.rows();
    using StackItem = std::tuple<const TreeType*, size_t, const TreeType*>;
    std::vector<StackItem> nodeStack;

    // First pass: compute sum of squared coefficients per level
    LevelMap levelNorms = getSumSquaredPerLevel(tree, tdata);

    // Convert to sqrt for normalization
    for (auto& [level, sumSquared] : levelNorms) {
      levelNorms[level] = std::sqrt(sumSquared);
    }

    // Start from the actual root of the tree (not tree.derived())
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
        // Store the full path of normalized sqrt(coefficients) and diameters
        // for this leaf
        leafData[currentNode] =
            std::make_pair(coefficientsStack, diametersStack);
      } else {
        // Push children (in reverse order to process leftmost child first)
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
  LevelMap max_coeff_per_level_;
  LevelMap sum_squared_per_level_;
};

}  // namespace FMCA

#endif  // FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H

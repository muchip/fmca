#ifndef FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H
#define FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "../FMCA/src/util/Macros.h"

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
    if (node.nsamplets() <= 0 || node.start_index() < 0 || node.start_index() + node.nsamplets() > tdata.size() ||
        node.block_size() <= 0) {
      return std::numeric_limits<Scalar>::quiet_NaN();  // Return NaN
    }
    auto segment = tdata.segment(node.start_index(), node.nsamplets());
    Scalar coefficient = segment.cwiseAbs().maxCoeff();
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
  // void traverseAndStackCoefficientsAndDiameters(const TreeType& tree,
  //                                               const Vector& tdata,
  //                                               LeafDataMap& leafData) {
  //   using StackItem = std::tuple<const TreeType*, size_t,
  //                                const TreeType*>;  // (node, level, parent)
  //   std::vector<StackItem> nodeStack;
  //   nodeStack.emplace_back(&tree.derived(), 0,
  //                          static_cast<const TreeType*>(nullptr));

  //   // Path stacks
  //   std::vector<Scalar> coefficientsStack;
  //   std::vector<Scalar> diametersStack;

  //   while (!nodeStack.empty()) {
  //     auto [currentNode, currentLevel, parentNode] = nodeStack.back();
  //     nodeStack.pop_back();

  //     // If we are returning from a deeper level, shrink stacks accordingly
  //     if (coefficientsStack.size() > currentLevel) {
  //       coefficientsStack.resize(currentLevel);
  //       diametersStack.resize(currentLevel);
  //     }

  //     // Compute the coefficient for this node
  //     std::optional<Scalar> coeffOpt = computeMaxCoeffNode(*currentNode,
  //     tdata); if (!coeffOpt) {
  //       // std::cout << "! Warning: Invalid node at level " << currentLevel
  //       // << std::endl;

  //       // If this is not the root and we're skipping this node,
  //       // mark parent as a leaf (if not already in the map)
  //       if (parentNode && leafData.find(parentNode) == leafData.end()) {
  //         auto parentCoeffs = coefficientsStack;
  //         auto parentDiams = diametersStack;
  //         leafData[parentNode] =
  //             std::make_pair(std::move(parentCoeffs),
  //             std::move(parentDiams));
  //       }
  //       continue;  // Skip children if no valid coefficient
  //     }

  //     // Compute the diameter as the norm of the last column of bb()
  //     Scalar diam = (currentNode->bb().col(2)).norm();

  //     coefficientsStack.push_back(*coeffOpt);
  //     diametersStack.push_back(diam);

  //     // Correctly evaluate these conditions
  //     bool noSamplets = (currentNode->nsamplets() <= 0);
  //     bool emptyBlock = (currentNode->block_size() <= 0);
  //     bool isLeaf = (currentNode->nSons() == 0);

  //     if (isLeaf) {
  //       // A real leaf => store the entire path
  //       leafData[currentNode] =
  //           std::make_pair(coefficientsStack, diametersStack);
  //     } else if (noSamplets || emptyBlock) {
  //       // Node has "no samplets" or is an empty block. Treat as a leaf
  //       std::cout << "! Warning: ";
  //       if (noSamplets) std::cout << "Node with no samplets";
  //       if (emptyBlock) std::cout << "Node with empty block";
  //       std::cout << " at level " << currentLevel << std::endl;

  //       // Store this node as a leaf
  //       leafData[currentNode] =
  //           std::make_pair(coefficientsStack, diametersStack);

  //       // We don't continue to children, as in your original logic
  //     } else {
  //       // Normal internal node => push children for DFS
  //       // Reverse order to maintain DFS ordering
  //       for (int i = currentNode->nSons() - 1; i >= 0; --i) {
  //         nodeStack.emplace_back(&currentNode->sons(i), currentLevel + 1,
  //                                currentNode);
  //       }
  //     }
  //   }

  //   // Add debugging to check coverage
  //   std::cout << "Found " << leafData.size() << " leaf nodes" << std::endl;
  // }

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
      Scalar diam = (currentNode->bb().col(2)).norm();  // currentNode->block_size() / N;  

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
 private:
  const TreeType* tree_ptr_ = nullptr;
  const Vector* tdata_ptr_ = nullptr;
  Index max_level_ = 0;
  LevelMap max_coeff_per_level_;
};

}  // namespace FMCA

#endif  // FMCA_EDGEDETECTION_SAMPLETCOEFFICIENTSANALYZER_H

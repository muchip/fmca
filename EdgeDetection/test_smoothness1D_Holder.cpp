#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ########################
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
#include "../FMCA/src/Clustering/ClusterTreeMetrics.h"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"
#include "../FMCA/src/util/Plotter.h"
#include "../FMCA/src/util/Tictoc.h"
#include "../FMCA/src/util/permutation.h"
#include "../TestPDE/read_files_txt.h"
#include "MultiGridFunctions.h"

#define DIM 1

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MatrixEvaluatorKernel =
    FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluatorKernel =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
using EigenCholesky =
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                          Eigen::MetisOrdering<int>>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////

Scalar f_Holder(Scalar x, Scalar alpha) {
  return pow(abs(x-0.3), alpha);
}


Vector eval_f_Holder(Matrix Points, Scalar alpha) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_Holder(Points(0, i), alpha);
  }
  return f;
}


//////////////////////////////////////////////////////////////////////////////////////////

template <typename Node>
Index computeMaxLevel(const Node& node) {
  Index max_level = node.level();  // Get the current node's level

  for (Index i = 0; i < node.nSons(); ++i) {
    max_level = std::max(max_level, computeMaxLevel(node.sons(i)));
  }

  return max_level;
}


template <typename Derived>
void printMaxCoefficientsPerLevel(const SampletTreeBase<Derived>& st,
                                  const Vector& tdata) {
  // Map to store the maximum coefficient for each level
  std::map<int, Scalar> levelMax;
  Vector levels;
  Vector values;

  for (const auto& node : st) {
    int level = node.level();
    Scalar coefficient =
        tdata.segment(node.start_index(), node.nsamplets()).norm();
    // Scalar coefficient = abs(tdata(node.block_id()));

    if (levelMax.find(level) == levelMax.end()) {
      levelMax[level] = coefficient;
    } else {
      levelMax[level] = std::max(levelMax[level], coefficient);
    }
  }

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

////////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
void printAverageCoefficientsPerLevel(const SampletTreeBase<Derived>& st,
                                      const Vector& tdata) {
  // Map to store the sum of coefficients and count for each level
  std::map<int, std::pair<Scalar, int>> levelSumAndCount;
  Vector levels;
  Vector averages;

  for (const auto& node : st) {
    int level = node.level();
    Scalar coefficient =
        tdata.segment(node.start_index(), node.nsamplets()).squaredNorm();
    // Scalar coefficient = abs(tdata(node.block_id()));

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

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

int main() {
  Tictoc T;
  Scalar alpha = 0.5;
  ///////////////////////////////// Inputs: points + maximum level
  Matrix P;
  readTXT("data/1D_100000_01.txt", P, DIM);
  ///////////////////////////////// Parameters
  const Scalar eta = 1. / DIM;
  const Index dtilde = 10;
  const Scalar threshold_kernel = 0;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = 2 * (dtilde - 1);
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P);

  Vector f_Holder = eval_f_Holder(P, alpha);
  Vector f_Holder_ordered = hst.toClusterOrder(f_Holder);
  Vector f_Holder_Samplets = hst.sampletTransform(f_Holder_ordered);

  ///////////////////////////////////////////////// Compute the maximum level

  Index max_level = computeMaxLevel(hst);
  std::cout << "Maximum level of the tree: " << max_level << std::endl;
  const Index nclusters = std::distance(hst.begin(), hst.end());
  std::cout << "Total number of clusters: " << nclusters << std::endl;
  printMaxCoefficientsPerLevel(hst, f_Holder_Samplets);
  Vector f_Holder_Samplets_natural_order = hst.toNaturalOrder(f_Holder_Samplets);
  saveVectorToFile(f_Holder_Samplets_natural_order,
                   "matlabPlots/f_Holder_Samplets.txt");

  return 0;
}

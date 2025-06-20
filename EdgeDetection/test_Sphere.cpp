#include <cmath>

#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
///////////////////////////
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/Samplets/adaptiveTreeSearch.h"
#include "read_files_txt.h"

#define DIM 3

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;

using namespace FMCA;

// Function to generate quasi-uniform points on a unit sphere using Fibonacci
// lattice
Matrix generateQuasiUniformPointsOnUnitSphere(int N) {
  Matrix P(DIM, N);

  // Golden ratio
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;

  for (int i = 0; i < N; ++i) {
    // Distribute points evenly along z-axis
    double z = 1.0 - (2.0 * i + 1.0) / N;

    // Radius at z
    double radius = std::sqrt(1.0 - z * z);

    // Golden angle increment
    double theta = 2.0 * M_PI * i / phi;

    // Convert to Cartesian coordinates
    P(0, i) = radius * std::cos(theta);  // x
    P(1, i) = radius * std::sin(theta);  // y
    P(2, i) = z;                         // z
  }

  return P;
}

// Function to compute function values: 0 if x > 0.3, 1 otherwise
Vector computeFunctionValues(const Matrix& P) {
  Vector f(P.cols());
  for (int i = 0; i < P.cols(); ++i) {
    f(i) = (P(0, i) + P(1, i) > 0.5) ? 0.0 : 1.0;
  }
  
  return f;
}
//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  // Number of points to generate
  const int N = 100000;

  // Generate quasi-uniform points on the unit sphere
  Matrix P = generateQuasiUniformPointsOnUnitSphere(N);
  std::cout << "Generated " << P.cols()
            << " quasi-uniform points on the unit sphere." << std::endl;
  std::cout << "Dimension P = " << P.rows() << " x " << P.cols() << std::endl;

  // Compute function values: 0 if x > 0.3, 1 otherwise
  Vector f = computeFunctionValues(P);
  std::cout << "Computed function values." << std::endl;
  std::cout << "Dimension f = " << f.rows() << " x " << f.cols() << std::endl;

  /////////////////////////////////

  const Scalar eta = 1. / DIM;
  const Index dtilde = 4;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const ST hst(samp_mom, 10, P);
  // const H2ST hst(mom, samp_mom, 0, P);

  // clusterTreeStatistics(hst, P);

  std::cout << "Tree done." << std::endl;

  Vector f_ordered = hst.toClusterOrder(f);
  Vector f_Samplets = hst.sampletTransform(f_ordered);

  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector tdata;
  tdata = f_Samplets;

  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(hst, tdata);

  Index max_level = coeff_analyzer.computeMaxLevel(hst);
  std::cout << "Max level = " << max_level << std::endl;

  auto max_coeff_per_level =
      coeff_analyzer.getMaxCoefficientsPerLevel(hst, tdata);
  for (auto [lvl, coeff] : max_coeff_per_level) {
    std::cout << "Level " << lvl << ": max coeff = " << coeff << "\n";
  }

  MapCoeffDiam leafData;
  // FMCA::SampletCoefficientsAnalyzer<ST>::LeafDataMap leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiameters(hst, tdata, leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-8);
  auto results = fitter.fitSlope();
  std::cout << "--------------------------------------------------------"
            << std::endl;

  //////////////////////////////////////////////////////////////////////////
  /*
    int count = 0;
    for (const auto& [leaf, res] : results) {
      if (res.get_slope() < 0) {
        std::cout << "Leaf bb: " << leaf->bb() << "\n";
        std::cout << "-------------------------" << std::endl;

        std::cout << "Slope: " << res.get_slope() << "\n";
        std::cout << "-------------------------" << std::endl;

        // Check if leaf exists in leafData
        auto it = leafData.find(leaf);
        if (it != leafData.end()) {
          const auto& coefficients =
              it->second.first;  // First vector = coefficients
          const auto& diameters = it->second.second;  // Second vector=diameters

          std::cout << "  Coefficients: ";
          for (const auto& coeff : coefficients) {
            std::cout << coeff << " ";
          }
          std::cout << "\n";
          std::cout << "-----------------------------------------" <<
          std::endl;

          std::cout << "\n  Diameters: ";
          for (const auto& diam : diameters) {
            std::cout << diam << " ";
          }
          std::cout << "\n";
          std::cout << "-------------------------" << std::endl;
        }
        std::cout << "=========================" << std::endl;
      }
      count++;
    }
*/
    for (const auto& [leaf, dataPair] : leafData) {
      auto coefficients = dataPair.first;
      auto diameters = dataPair.second;
      for (int j = 0; j < leaf->block_size(); ++j) {
        if (leaf->indices()[j] == 8768) {
          std::cout << "leaf bb " << leaf->bb() << std::endl;
          // coefficients is likely a vector or array, need to print specific
          // elements
          std::cout << "coeffs ";
          for (const auto& coeff : coefficients) {
            std::cout << coeff << " ";
          }
          std::cout << std::endl;
          std::cout << "diams ";
          for (const auto& diam : diameters) {
            std::cout << diam << " ";
          }
          std::cout << std::endl;
        }
      }
    }

    for (const auto& [leaf, res] : results) {
      for (int j = 0; j < leaf->block_size(); ++j) {
        if (leaf->indices()[j] == 8768) {
          std::cout << res.get_slope() << std::endl;
        }
      }
    }

/*
  FMCA::Scalar norm2 = tdata.squaredNorm();
  std::vector<const ST*> adaptive_tree =
      adaptiveTreeSearch(hst, tdata, 0 * norm2);
  const FMCA::Index nclusters = std::distance(hst.begin(), hst.end());

  FMCA::Vector thres_tdata = tdata;
  thres_tdata.setZero();
  FMCA::Index nnz = 0;
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const ST& node = *(adaptive_tree[i]);
      const FMCA::Index ndist =
          node.is_root() ? node.Q().cols() : node.nsamplets();
      thres_tdata.segment(node.start_index(), ndist) =
          tdata.segment(node.start_index(), ndist);
      nnz += ndist;
    }
  }
  std::cout << "active coefficients: " << nnz << " / " << P.cols() << std::endl;
  std::cout << "tree error: " << (thres_tdata - tdata).norm() / tdata.norm()
            << std::endl;

  std::vector<bool> check_id(P.cols());
  for (FMCA::Index i = 0; i < adaptive_tree.size(); ++i) {
    if (adaptive_tree[i] != nullptr) {
      const ST& node = *(adaptive_tree[i]);
      if (!node.nSons() || adaptive_tree[node.sons(0).block_id()] == nullptr)
        for (FMCA::Index j = 0; j < node.block_size(); ++j)
          check_id[node.indices()[j]] = true;
    }
  }
  for (FMCA::Index i = 0; i < check_id.size(); ++i) {
    if (!check_id[i]) {
      std::cerr << "missing index in adaptive tree";
      return 1;
    }
  }

  std::vector<FMCA::Matrix> bbvec;
  int counter1 = 0;
  int counter2 = 0;
  for (const auto* it : adaptive_tree) {
    counter1++;
    // First check if the pointer is not null
    if (it != nullptr &&
        (!it->nSons() ||
         (it->nSons() > 0 && it->sons(0).block_id() < adaptive_tree.size() &&
          adaptive_tree[it->sons(0).block_id()] == nullptr))) {
      // Get the bounding box
      FMCA::Matrix bb = it->bb();

      // Check if the bounding box contains any infinite values
      bool hasInfiniteValues = false;
      for (int i = 0; i < bb.rows(); ++i) {
        for (int j = 0; j < bb.cols(); ++j) {
          if (!std::isfinite(bb(i, j))) {
            // print the bounding box 
            std::cout << "bb " << it->bb() << std::endl;
            std::cout << "infinite bb has size = " << it->block_size() << std::endl;
            hasInfiniteValues = true;
            break;
          }
        }
        if (hasInfiniteValues) break;
      }

      // Only add bounding box if it doesn't contain infinite values
      if (!hasInfiniteValues) {
        // Add bounding box check for each point in the node
        for (auto j = 0; j < it->block_size(); ++j) {
          if (!FMCA::internal::inBoundingBox(*it, P.col(it->indices()[j]))) {
            std::cerr << "Warning: point outside leaf bounding box"
                      << std::endl;
            // Don't assert, just warn
          }
        }

        bbvec.push_back(bb);
        counter2++;
      } else {
        std::cerr << "Warning: Skipping bounding box with infinite values"
                  << std::endl;
      }
    }
  }
  std::cout << counter1 << " " << counter2 << std::endl;

  FMCA::IO::plotBoxes("AdaptiveTree.vtk", bbvec);
  FMCA::IO::plotPointsColor("AdaptiveTreeFunction.vtk", P, f);

*/
  //////////////////////////////////////////////////////////////////////////
  // Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = res.get_slope();
      Scalar dtilde_binned = std::abs(std::floor((slope + 0.25) / 0.5) * 0.5);
      colr(leaf->indices()[j]) = slope;
    }
  }

  FMCA::IO::plotPointsColor("Slopes_sphere.vtk", P, colr);
  FMCA::IO::plotPointsColor("Points_sphere.vtk", P, f);

  // // Save the points and function values to files
  // std::ofstream points_file("points.txt");
  // for (int i = 0; i < P.cols(); ++i) {
  //     for (int j = 0; j < P.rows(); ++j) {
  //         points_file << P(j, i) << " ";
  //     }
  //     points_file << "\n";
  // }
  // points_file.close();

  // std::ofstream function_file("function.txt");
  // for (int i = 0; i < f.rows(); ++i) {
  //     function_file << f(i) << "\n";
  // }
  // function_file.close();

  return 0;
}
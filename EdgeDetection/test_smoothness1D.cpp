#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ########################
#include "SmoothnessDetection.h"

#define DIM 1
using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
Scalar f_jump(Scalar x, Scalar step, Scalar jump) {
  return (x < step) ? 1 : (1 + jump);
}

Scalar f_wave(Scalar x, Scalar freq) {
  return (x < 0.3) ? std::sin(freq * 0.3) : (std::sin(freq * x));
}

Scalar f_abs3(Scalar x, Scalar step) {
  return (x - step) * (x - step) * abs(x - step);
}

Scalar f_elaborated(Scalar x) {
  if (x < -0.4) {
    return 6;
  } else if (x >= -0.4 && x < -0.35) {
    return 0.1 * std::abs(-20 * x - 9) + 6;
  } else if (x >= -0.35 && x < -0.15) {
    return 0.1 * std::abs(-20 * x - 5) + 6;
  } else if (x >= -0.15 && x < -0.05) {
    return 0.1 * std::abs(-20 * x - 1) + 6;
  } else if (x >= -0.05 && x < 0.55) {
    return 6 + std::sin(20 * FMCA_PI * x);
  } else if (x >= 0.55) {
    return 0.2 * std::sin(6 * FMCA_PI * x);
  }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
Vector eval_f_jump(Matrix Points, Scalar step, Scalar jump) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_jump(Points(0, i), step, jump);
  }
  return f;
}

Vector eval_f_wave(Matrix Points, Scalar freq) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_wave(Points(0, i), freq);
  }
  return f;
}

Vector eval_f_abs3(Matrix Points, Scalar step) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_abs3(Points(0, i), step);
  }
  return f;
}

Vector eval_f_elaborated(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_elaborated(Points(0, i));
  }
  return f;
}

//////////////////////////////////////////////////////////////////////////////////////////

int main() {
  Tictoc T;
  Scalar threshold_active_leaves = 1e-20;
  Scalar step = 0.3;
  Scalar jump = 1;
  Scalar freq = 20;
  /////////////////////////////////
  Matrix P;
  readTXT("data/1D_10000.txt", P, DIM);
  /////////////////////////////////
  const std::string function_type = "f_elaborated";
  const Scalar eta = 1. / DIM;
  const Index dtilde = 10;
  const Scalar threshold_kernel = 0;
  const Scalar threshold_weights = 0;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "threshold_kernel    " << threshold_kernel << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  const H2SampletTree<RandomProjectionTree> hst(mom, samp_mom, 0, P);


  Vector f_jump = eval_f_jump(P, step, jump);
  Vector f_jump_ordered = hst.toClusterOrder(f_jump);
  Vector f_jump_Samplets = hst.sampletTransform(f_jump_ordered);

  Vector f_wave = eval_f_wave(P, freq);
  Vector f_wave_ordered = hst.toClusterOrder(f_wave);
  Vector f_wave_Samplets = hst.sampletTransform(f_wave_ordered);

  Vector f_3 = eval_f_abs3(P, step);
  Vector f_3_ordered = hst.toClusterOrder(f_3);
  Vector f_3_Samplets = hst.sampletTransform(f_3_ordered);

  Vector f_elaborated = eval_f_elaborated(P);
  Vector f_elaborated_ordered = hst.toClusterOrder(f_elaborated);
  Vector f_elaborated_Samplets = hst.sampletTransform(f_elaborated_ordered);
  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector tdata;
  if (function_type == "jump") {
    tdata = f_jump_Samplets;
  } else if (function_type == "wave") {
    tdata = f_wave_Samplets;
  } else if (function_type == "f_3") {
    tdata = f_3_Samplets;
  } else if (function_type == "f_elaborated") {
    tdata = f_elaborated_Samplets;
  }

  Index max_level = computeMaxLevel(hst);
  std::cout << "Maximum level of the tree: " << max_level << std::endl;
  const Index nclusters = std::distance(hst.begin(), hst.end());
  std::cout << "Total number of clusters: " << nclusters << std::endl;

  // Coeff decay
  printMaxCoefficientsPerLevel(hst, tdata);
  // Singularity localization
  std::vector<const H2SampletTree<RandomProjectionTree>*> adaptive_tree =
      computeAdaptiveTree(hst, tdata, threshold_active_leaves);
  std::vector<const H2SampletTree<RandomProjectionTree>*> nodes;
  std::vector<FMCA::Matrix> bbvec_active;
  computeNodeAndBBActive(adaptive_tree, nodes, bbvec_active);

  // for (const auto& matrix : bb_vec_active) {
  //   std::cout << "Matrix " << std::endl;
  //   std::cout << matrix << std::endl;
  // }

  printIntervalsPython(bbvec_active);
  // for (const H2SampletTree<ClusterTree>* node : nodes) {
  //   if (node != nullptr) {
  //     // Call printMaxCoefficientsPerBranch for each leaf node
  //     std::cout << "----------------------------------------" << std::endl;
  //     std::cout << "Node bb = (" << node->bb() << ")" << std::endl;
  //     printMaxCoefficientsPerBranch(node, tdata);
  //   }
  // }

  return 0;
}

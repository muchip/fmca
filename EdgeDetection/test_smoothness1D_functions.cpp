#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
// ########################
#include "FitDtilde.h"
#include "InputOutput.h"
#include "Regression.h"
#include "SmoothnessDetection.h"

#define DIM 1

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using H2ST = FMCA::H2SampletTree<FMCA::ClusterTree>;

using namespace FMCA;

//////////////////////////////////////////////////////////////////////////////////////////
Scalar f_smooth(Scalar x) {
  // return std::sin(x);
  // return std::exp(-x*x);
  return std::sqrt(2 - x * x);
}
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
    return 0.5 * std::abs(-20 * x - 9) + 6;
  } else if (x >= -0.35 && x < -0.15) {
    return 0.5 * std::abs(-20 * x - 5) + 6;
  } else if (x >= -0.15 && x < -0.05) {
    return 0.5 * std::abs(-20 * x - 1) + 6;
  } else if (x >= -0.05 && x < 0.55) {
    return 6 + std::sin(20 * FMCA_PI * x);
  } else if (x >= 0.55) {
    return 0.2 * std::sin(6 * FMCA_PI * x);
  }
  return 0;
}

// Scalar f_elaboratedC1(Scalar x) {
//   if (x < -0.6) {
//     return 5. * 0. + 2.;
//   } else if (x >= -0.6 && x < -0.2) {
//     return 5. * (-x - 6. / 10) + 2.;
//   } else if (x >= -0.2 && x < 0.2) {
//     return 5. * (10. * x * std::abs(x)) + 2.;
//   } else if (x >= 0.2 && x < 0.6) {
//     return 5. * (-x + 6. / 10) + 2.;
//   } else if (x >= 0.6) {
//     return 5. * 0. + 2.;
//   }
//   return 0;
// }

Scalar f_elaboratedC1(Scalar x) {
  if (x < -0.4) {
    return 6.0;
  } else if (x < -0.35) {  // implies x >= -0.4
    return 0.5 * std::abs(-20.0 * x - 9.0) + 6.0;
  } else if (x < -0.15) {  // implies x >= -0.35
    return 0.5 * std::abs(-20.0 * x - 5.0) + 6.0;
  } else if (x < -0.05) {  // implies x >= -0.15
    return 0.5 * std::abs(-20.0 * x - 1.0) + 6.0;
  } else if (x < 0.55) {  // implies x >= -0.05
    return 6.0 + std::sin(20.0 * FMCA_PI * x);
  } else if (x < 0.7) {  // implies x >= 0.55
    Scalar d = x - 0.7;
    return 4.0 + 20.0 * d * d;
  } else {  // x >= 0.7
    Scalar d = x - 0.7;
    return 4.0 - 20.0 * d * d;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
Vector eval_f_smooth(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_smooth(Points(0, i));
  }
  return f;
}
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

Vector eval_f_elaboratedC1(Matrix Points) {
  Vector f(Points.cols());
  for (Index i = 0; i < Points.cols(); ++i) {
    f(i) = f_elaboratedC1(Points(0, i));
  }
  return f;
}

//////////////////////////////////////////////////////////////////////////////////////////

int main() {
  Scalar threshold_active_leaves = 0;
  Scalar step = 0.3;
  Scalar jump = 1;
  Scalar freq = 20;
  /////////////////////////////////
  Matrix P;
  readTXT("data/1D_4M.txt", P, DIM);
  const std::string outputFile_step = "Plots/output_step1.txt";
  /////////////////////////////////
  const std::string function_type = "f_elaboratedC1";
  const Scalar eta = 1. / DIM;
  const Index dtilde = 5;
  const Scalar mpole_deg = (dtilde != 1) ? 2 * (dtilde - 1) : 2;
  std::cout << "eta                 " << eta << std::endl;
  std::cout << "dtilde              " << dtilde << std::endl;
  std::cout << "mpole_deg           " << mpole_deg << std::endl;

  const Moments mom(P, mpole_deg);
  const SampletMoments samp_mom(P, dtilde - 1);
  // const H2SampletTree<ClusterTree> hst(mom, samp_mom, 0, P);
  const ST hst(samp_mom, 0, P);
  // const H2ST hst(mom, samp_mom, 0, P);

  Vector f_smooth = eval_f_smooth(P);
  Vector f_smooth_ordered = hst.toClusterOrder(f_smooth);
  Vector f_smooth_Samplets = hst.sampletTransform(f_smooth_ordered);

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

  Vector f_elaboratedC1 = eval_f_elaboratedC1(P);
  Vector f_elaboratedC1_ordered = hst.toClusterOrder(f_elaboratedC1);
  Vector f_elaboratedC1_Samplets = hst.sampletTransform(f_elaboratedC1_ordered);
  ///////////////////////////////////////////////// Compute the decay of the
  /// coeffcients
  Vector tdata;
  Vector f;
  if (function_type == "jump") {
    f = f_jump;
    tdata = f_jump_Samplets;
  } else if (function_type == "f_smooth") {
    f = f_smooth;
    tdata = f_smooth_Samplets;
  } else if (function_type == "wave") {
    f = f_wave;
    tdata = f_wave_Samplets;
  } else if (function_type == "f_3") {
    f = f_3;
    tdata = f_3_Samplets;
  } else if (function_type == "f_elaborated") {
    f = f_elaborated;
    tdata = f_elaborated_Samplets;
  } else if (function_type == "f_elaboratedC1") {
    f = f_elaboratedC1;
    tdata = f_elaboratedC1_Samplets;
  }

  Index max_level = computeMaxLevel(hst);
  std::cout << "Maximum level of the tree: " << max_level << std::endl;
  const Index nclusters = std::distance(hst.begin(), hst.end());
  std::cout << "Total number of clusters: " << nclusters << std::endl;

  std::map<const ST*, std::pair<std::vector<Scalar>, std::vector<Scalar>>>
      leafData;
  traverseAndStackCoefficientsAndDiameters(hst, tdata, leafData);

  auto results = FitDtilde(leafData, dtilde, 1e-16);
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;

  ////////////////////////////////////////////////////////////////////////////
  std::vector<Scalar> x;       // Bounding box min x-coordinates
  std::vector<Scalar> coeffs;  // Slopes duplicated for step function

  for (const auto& [leaf, res] : results) {
    const auto& bbox = leaf->bb();  // Assuming bb() gives the bounding box
    Scalar minX = bbox(0);          // Minimum x-coordinate
    Scalar maxX = bbox(1);          // Maximum x-coordinate

    // Append points for the step function
    x.push_back(minX);
    coeffs.push_back(res.get_dtilde());
  }

  // Write the data to a file
  std::ofstream file("/Users/saraavesani/Desktop/Archive/points_with_color.py");
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
  std::cout << "Data written to " << "points_with_color.py" << std::endl;

  //   for (const auto& [leaf, dataPair] : leafData) {
  //     // if (count >= 20) break;
  //     auto coefficients = dataPair.first;  // First vector = coefficients
  //     const auto& diameters = dataPair.second;     // Second vector =
  //     diameters if (leaf->bb()(0, 0) > 0.099 && leaf->bb()(0, 1) < 0.101) {

  //     std::cout << "Leaf bb: " << leaf->bb() << "\n";
  //     std::cout << "-----------------------------------------" << std::endl;
  //     size_t n = coefficients.size();

  //     Scalar accum = 0.;
  //     for (int i = 0; i < n; ++i) {
  //         accum += coefficients[i] * coefficients[i];
  //     }
  //     Scalar norm = sqrt(accum);
  //     if (norm != 0) {
  //       for (auto& coeff : coefficients) {
  //           coeff /= norm;
  //       }
  //     }

  //     std::cout << "  Coefficients: ";
  //     int zero_count = 0;
  //     for (const auto& coeff : coefficients) {
  //         std::cout <<  coeff << " ";
  //         if (coeff == 0) {
  //             zero_count++;
  //         }
  //     }
  //     // std::cout << "\n";
  //     // std::cout << "number of 0 coeffs = " << zero_count << "\n";
  //     std::cout << "\n";
  //     std::cout << "-----------------------------------------" << std::endl;

  //     std::cout << "\n  Diameters: ";
  //     for (const auto& diam : diameters) {
  //         std::cout << diam << " ";
  //     }
  //     std::cout << "\n";
  //     std::cout << "-------------------------" << std::endl;
  //          }
  // }

  // Coeff decay
  // printMaxCoefficientsPerLevel(hst, tdata);

  return 0;
}

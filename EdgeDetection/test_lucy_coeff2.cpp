#include <iostream>
#include <random>

#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Tictoc.h"
#include "SampletCoefficientsAnalyzer.h"
#include "SlopeFitter.h"
#include "read_files_txt.h"

#define DIM 3

using ST = FMCA::SampletTree<FMCA::ClusterTree>;
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MapCoeffDiam =
    std::map<const ST*,
             std::pair<std::vector<FMCA::Scalar>, std::vector<FMCA::Scalar>>>;
using namespace FMCA;

/////////////////////////////////////////////// BB
void computeBoundingBox(const Matrix& points, Scalar& xmin, Scalar& xmax,
                        Scalar& ymin, Scalar& ymax, Scalar& zmin,
                        Scalar& zmax) {
  xmin = xmax = points(0, 0);
  ymin = ymax = points(1, 0);
  zmin = zmax = points(2, 0);

  for (Index i = 1; i < points.cols(); ++i) {
    xmin = std::min(xmin, points(0, i));
    xmax = std::max(xmax, points(0, i));
    ymin = std::min(ymin, points(1, i));
    ymax = std::max(ymax, points(1, i));
    zmin = std::min(zmin, points(2, i));
    zmax = std::max(zmax, points(2, i));
  }
}

/////////////////////////////////////////////// CORNER
// Vector createCornerFunction(const Matrix& points) {
//   Vector data(points.cols());
//   Scalar xmin, xmax, ymin, ymax, zmin, zmax;
//   computeBoundingBox(points, xmin, xmax, ymin, ymax, zmin, zmax);

//   // Expand bounding box slightly
//   Scalar margin = 0.1;
//   Scalar dx = (xmax - xmin) * margin;
//   Scalar dy = (ymax - ymin) * margin;
//   Scalar dz = (zmax - zmin) * margin;

//   xmin -= dx;
//   xmax += dx;
//   ymin -= dy;
//   ymax += dy;
//   zmin -= dz;
//   zmax += dz;

//   // Create a non-axis aligned line direction in 3D
//   Vector pt(3);
//   pt.setOnes();
//   pt /= std::sqrt(3.0);  // Normalize to unit vector

//   for (Index i = 0; i < points.cols(); ++i) {
//     Scalar x = points(0, i);
//     Scalar y = points(1, i);
//     Scalar z = points(2, i);

//     // Normalize coordinates to [-1, 1] cube
//     Scalar x_norm = 2.0 * (x - xmin) / (xmax - xmin) - 1.0;
//     Scalar y_norm = 2.0 * (y - ymin) / (ymax - ymin) - 1.0;
//     Scalar z_norm = 2.0 * (z - zmin) / (zmax - zmin) - 1.0;

//     // Create normalized point vector
//     Vector point(3);
//     point(0) = x_norm;
//     point(1) = y_norm;
//     point(2) = z_norm;
//     Scalar signed_dist = point.dot(pt) - 0.5 * std::sqrt(3.0);
//     data(i) = std::abs(signed_dist);
//   }
//   return data;
// }
Vector createCornerFunction(const Matrix& points) {
  Vector data(points.cols());

  for (Index i = 0; i < points.cols(); ++i) {
    Scalar x = points(0, i);
    Scalar y = points(1, i);
    Scalar z = points(2, i);

    // Create a corner discontinuity at x = 0.4
    // The corner function is the distance from the plane x = 0.4
    data(i) = 10*std::abs(x - 0.4);
  }
  return data;
}

/////////////////////////////////////////////// JUMP
Vector createJumpFunction(const Matrix& points) {
  Vector data(points.cols());

  Scalar xmin, xmax, ymin, ymax, zmin, zmax;
  computeBoundingBox(points, xmin, xmax, ymin, ymax, zmin, zmax);

  // Expand bounding box slightly
  Scalar margin = 0.1;
  Scalar dx = (xmax - xmin) * margin;
  Scalar dy = (ymax - ymin) * margin;
  Scalar dz = (zmax - zmin) * margin;

  xmin -= dx;
  xmax += dx;
  ymin -= dy;
  ymax += dy;
  zmin -= dz;
  zmax += dz;

  for (Index i = 0; i < points.cols(); ++i) {
    Scalar x = points(0, i);
    Scalar y = points(1, i);
    Scalar z = points(2, i);

    // Normalize coordinates to [-1, 1] cube
    Scalar x_norm = 2.0 * (x - xmin) / (xmax - xmin) - 1.0;
    Scalar y_norm = 2.0 * (y - ymin) / (ymax - ymin) - 1.0;
    Scalar z_norm = 2.0 * (z - zmin) / (zmax - zmin) - 1.0;

    // Initialize jump value
    Scalar jump_value = 0.0;

    // // Jump plane 1: diagonal plane
    if (x_norm + y_norm + z_norm > 0.001) {
      jump_value += 1.0;
    }

    // Jump plane: curved surface
    // if (x_norm * x_norm + y_norm * y_norm - z_norm > 0.3) {
    //   jump_value -= 0.8;
    // }
    data(i) = jump_value;
  }

  return data;
}

//////////////////////////////////////////////////////////////////////////////////////////
int main() {
  constexpr FMCA::Index dtilde = 5;
  FMCA::Tictoc T;
  FMCA::Matrix P;
  readTXT("data/lucy4Nested_level5.txt", P, 3);
  Scalar Nd = P.cols();
  FMCA::Vector data1;
  data1 = createJumpFunction(P);
  FMCA::Matrix P_corner;
  std::vector<Index> corner_indices;

  Index corner_count = 0;
  for (Index i = 0; i < data1.size(); ++i) {
    if (data1(i) == 1) {
      corner_count++;
    }
  }

  P_corner.resize(P.rows(), corner_count);

  Index corner_idx = 0;
  for (Index i = 0; i < data1.size(); ++i) {
    if (data1(i) == 1) {
      P_corner.col(corner_idx) = P.col(i);
      corner_indices.push_back(i);
      corner_idx++;
    }
  }

  FMCA::Vector data2_corner;
  data2_corner = createCornerFunction(P_corner);

  FMCA::Vector data2 = FMCA::Vector::Zero(data1.size());

  for (Index i = 0; i < corner_indices.size(); ++i) {
    data2(corner_indices[i]) = data2_corner(i);
  }

  FMCA::Vector data = data1;
  for (Index i = 0; i < data1.size(); ++i) {
    if (data1(i) != 0) {
      data(i) = data2(i);
    } else {
      data(i) = 3.0;
    }
  
  }

  FMCA::IO::plotPointsColor("FunctionLucy.vtk", P, data);
  /////////////////////////////////
  T.tic();
  const SampletMoments samp_mom(P, dtilde - 1);
  ST st(samp_mom, 0, P);
  T.toc("samplet tree generation:");
  const FMCA::Vector scoeffs = st.sampletTransform(st.toClusterOrder(data));

  ///////////////////////////////////////////////// Compute the global decay of
  /// the
  /// coeffcients
  T.tic();
  SampletCoefficientsAnalyzer<ST> coeff_analyzer;
  coeff_analyzer.init(st, scoeffs);

  Index max_level = coeff_analyzer.computeMaxLevel(st);
  std::cout << "Max level = " << max_level << std::endl;

  auto max_coeff_per_level =
      coeff_analyzer.getMaxCoefficientsPerLevel(st, scoeffs);

  for (FMCA::Index i = 1; i < max_coeff_per_level.size(); ++i)
    std::cout << "c=" << max_coeff_per_level[i] / std::sqrt(Nd) << " alpha="
              << std::log(max_coeff_per_level[i - 1] / max_coeff_per_level[i]) /
                     (std::log(2))
              << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;

  auto squared_coeff_per_level =
      coeff_analyzer.getSumSquaredPerLevel(st, scoeffs);
  std::cout << "--------------------------------------------------------"
            << std::endl;

  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) << std::endl;

  std::cout << "--------------------------------------------------------"
            << std::endl;
  for (FMCA::Index i = 1; i < squared_coeff_per_level.size(); ++i)
    std::cout << "c=" << std::sqrt(squared_coeff_per_level[i]) / std::sqrt(Nd)
              << " alpha="
              << std::log(std::sqrt(squared_coeff_per_level[i - 1]) /
                          std::sqrt(squared_coeff_per_level[i])) /
                     (std::log(2))
              << std::endl;

  //////////////////////////////////////////////////////////////

  MapCoeffDiam leafData;
  coeff_analyzer.traverseAndStackCoefficientsAndDiametersL2Norm(st, scoeffs,
                                                                leafData);

  SlopeFitter<ST> fitter;
  fitter.init(leafData, dtilde, 1e-12);
  auto results = fitter.fitSlope();
  std::cout << "--------------------------------------------------------"
            << std::endl;
  T.toc("Slope Fitting: ");

  // Initialize color vector for each column in P
  Vector colr(P.cols());
  for (const auto& [leaf, res] : results) {
    for (int j = 0; j < leaf->block_size(); ++j) {
      Scalar slope = res.get_slope();
      // save the slope just if slope < 2, otherwirse save dtilde
      // Scalar slope_filtered = (slope <= 1.5) ? slope : dtilde;
      // Scalar dtilde_binned = std::abs(std::floor((slope + 0.25) / 0.5) * 0.5); 
      colr(leaf->indices()[j]) = slope;
    }
  }
  // Print the min value of colr
  std::cout << "Min value of colr: " << colr.minCoeff() << std::endl;

  FMCA::IO::plotPointsColor("SlopeLucy.vtk", P, colr);
  // FMCA::IO::plotPointsColor("FunctionSphere.vtk", P, data);

  return 0;
}
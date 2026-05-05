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

#ifndef FMCA_EDGEDETECTION_SLOPEFITTER_H
#define FMCA_EDGEDETECTION_SLOPEFITTER_H

namespace FMCA {
template <typename Scalar>
struct FitResult {
 private:
  Scalar slope;
  Scalar constant;
  Scalar R2;

 public:
  FitResult(Scalar get_slope = 0, Scalar get_constant = 0, Scalar get_R2 = 0)
      : slope(get_slope), constant(get_constant), R2(get_R2) {}

  Scalar get_slope() const { return slope; }
  Scalar get_constant() const { return constant; }
  Scalar get_R2() const { return R2; }
};

//////////////////////////////////////////////////////////////////////////////
/**
 * @brief Performs slope fitting for tree-structured data.
 *
 * @tparam TreeType Type of the tree structure being analyzed.
 *
 * This class fits linear models in log-log space to coefficient decay data
 * associated with tree leaves. It supports both QR decomposition and weighted
 * regression methods, with automatic handling of edge cases such as smooth
 * decay and insufficient data.
 */
template <typename TreeType>
class SlopeFitter {
 public:
  using LeafDataMap =
      std::map<const TreeType*,
               std::pair<std::vector<Scalar>, std::vector<Scalar>>>;
  using ResultMap = std::map<const TreeType*, FitResult<Scalar>>;

  SlopeFitter() {}

  SlopeFitter(const LeafDataMap& leafData, Scalar dtilde,
              Scalar smallThreshold) {
    init(leafData, dtilde, smallThreshold);
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
  * @brief Initializes the SlopeFitter with data and parameters.
  *
  * @param leaf_data Map of leaf pointers to (coefficients, diameters) pairs.
  * @param dtilde Upper bound for the fitted slope (set to 1 if ≤ 0).
  * @param small_threshold Threshold for smooth decay detection (set to
  *                        FMCA_ZERO_TOLERANCE if < 0).
  */
  void init(const LeafDataMap& leaf_data, Scalar dtilde,
            Scalar small_threshold) {
    leaf_data_ = leaf_data;
    dtilde_ = dtilde > 0 ? dtilde : 1;
    small_threshold_ =
        small_threshold >= 0 ? small_threshold : FMCA_ZERO_TOLERANCE;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
  * @brief Fits slopes using QR decomposition.
  *
  * Performs least-squares fitting in log-log space using Householder QR
  * decomposition. Handles edge cases:
  * - Returns dtilde if fewer than 2 data points remain after filtering.
  * - Returns dtilde if the last two coefficients indicate smooth decay.
  *
  * @return Map of leaf pointers to FitResult containing slope, intercept,
  *         and R² values.
  */
  ResultMap fitSlope() {
    ResultMap results;
    for (const auto& [leaf, dataPair] : leaf_data_) {
      auto coefficients = dataPair.first;
      auto diameters = dataPair.second;
      size_t n = coefficients.size() - 1;
      std::vector<Scalar> filtered_coefficients;
      std::vector<Scalar> filtered_diameters;

      for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(coefficients[i])) {
          filtered_coefficients.push_back(coefficients[i]);
          filtered_diameters.push_back(diameters[i]);
        }
      }
      size_t filtered_n = filtered_coefficients.size();
      Scalar norm = std::sqrt(std::accumulate(
          filtered_coefficients.begin(), filtered_coefficients.end(), Scalar(0),
          [](Scalar sum, Scalar c) { return sum + c * c; }));
      // Edge case 1: Too few coefficients
      if (filtered_n < 2) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0, 1.0);
      }
      // Edge case 2: Smooth decay, last two coefficients are near zero
      else if (std::abs(filtered_coefficients[filtered_n - 1] / norm) <
                   small_threshold_ &&
               std::abs(filtered_coefficients[filtered_n - 2] / norm) <
                   small_threshold_) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0, 1.0);
      }

      else {
        // Solve Ax = b using QR decomposition
        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> A(filtered_n, 2);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(filtered_n);

        for (size_t i = 0; i < filtered_n; ++i) {
          A(i, 0) = 1.0;
          A(i, 1) = (filtered_diameters[i] == 0)
                        ? std::log(FMCA_ZERO_TOLERANCE)
                        : std::log(std::abs(
                              filtered_diameters[i]));
          b(i) = (filtered_coefficients[i] == 0)
                     ? std::log(FMCA_ZERO_TOLERANCE)
                     : std::log(std::abs(filtered_coefficients[i]));
        }

        // Compute the least squares solution using the QR decomposition
        Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 2>> qr(A);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> Q =
            qr.householderQ() *
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
                A.rows(), 2);
        Eigen::Matrix<Scalar, 2, 2> R =
            qr.matrixQR()
                .topLeftCorner(2, 2)
                .template triangularView<Eigen::Upper>();
        // y = Q^T * b
        Eigen::Matrix<Scalar, 2, 1> y = Q.transpose() * b;
        // R x = y
        Eigen::Matrix<Scalar, 2, 1> x =
            R.template triangularView<Eigen::Upper>().solve(y);
        // Compute R2
        Scalar mean_b = b.mean();
        Scalar ss_tot = (b.array() - mean_b).square().sum();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_pred = A * x;
        Scalar ss_res = (b - b_pred).squaredNorm();
        Scalar r_squared = (ss_tot > FMCA_ZERO_TOLERANCE) ?
                           1.0 - (ss_res / ss_tot) : 1.0;
        // x(0) = intercept, x(1) = slope
        results[leaf] = FitResult<Scalar>(std::min(dtilde_, x(1)), x(0), r_squared);
      }
    }
    return results;
  }

  //////////////////////////////////////////////////
  /**
 * @brief Fits slopes using weighted or unweighted linear regression.
 *
 * Performs least-squares fitting in log-log space using direct calculation
 * of regression coefficients. When weighted, applies inverse distance-squared
 * weighting to emphasize points near the mean.
 *
 * @param weighted If true, applies distance-based weighting (default: false).
 * @return Map of leaf pointers to FitResult containing slope, intercept,
 *         and R² values.
 */
  ResultMap fitSlopeRegression(bool weighted = false) {
    ResultMap results;

    for (const auto& [leaf, dataPair] : leaf_data_) {
      auto coefficients = dataPair.first;
      const auto& diameters = dataPair.second;
      size_t n = coefficients.size();
      std::vector<Scalar> filtered_coefficients;
      std::vector<Scalar> filtered_diameters;
      for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(coefficients[i])) {
          filtered_coefficients.push_back(coefficients[i]);
          filtered_diameters.push_back(diameters[i]);
        }
      }
      size_t filtered_n = filtered_coefficients.size();
      Scalar norm = std::sqrt(std::accumulate(
          filtered_coefficients.begin(), filtered_coefficients.end(), Scalar(0),
          [](Scalar sum, Scalar c) { return sum + c * c; }));
      // Edge case 1: Too few coefficients
      if (filtered_n < 2) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0, 1.0);
      }
      // Edge case 2: Smooth decay, last two coefficients are near zero
      else if (std::abs(filtered_coefficients[filtered_n - 1] / norm) <
                   small_threshold_ &&
               std::abs(filtered_coefficients[filtered_n - 2] / norm) <
                   small_threshold_) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0, 1.0);
      } else {
        std::vector<Scalar> x_values, y_values;
        x_values.reserve(filtered_n);
        y_values.reserve(filtered_n);
        for (size_t i = 0; i < filtered_n; ++i) {
          auto coef =
              std::max(std::abs(filtered_coefficients[i]), FMCA_ZERO_TOLERANCE);
          x_values.emplace_back(std::log(filtered_diameters[i]));
          y_values.emplace_back(std::log(coef));
        }
        Scalar mean_x =
            std::accumulate(x_values.begin(), x_values.end(), Scalar(0)) /
            filtered_n;
        Scalar mean_y =
            std::accumulate(y_values.begin(), y_values.end(), Scalar(0)) /
            filtered_n;
        // weihghts
        std::vector<Scalar> weights(filtered_n, Scalar(1.0));
        if (weighted) {
          Scalar sum_squared_diff = 0.0;
          for (size_t i = 0; i < filtered_n; ++i) {
            Scalar diff = x_values[i] - mean_x;
            sum_squared_diff += diff * diff;
          }
          for (size_t i = 0; i < filtered_n; ++i) {
            Scalar dist_x = (x_values[i] - mean_x);
            Scalar weight = 1.0 / (1.0 + dist_x * dist_x);
            weights[i] = weight;
          }
          Scalar sum_weights =
              std::accumulate(weights.begin(), weights.end(), Scalar(0));
          if (sum_weights > FMCA_ZERO_TOLERANCE) {
            Scalar scale = Scalar(filtered_n) / sum_weights;
            for (auto& w : weights) w *= scale;
          }
        }
        // slope and intercept
        Scalar numerator = 0.0;
        Scalar denominator = 0.0;
        for (size_t i = 0; i < filtered_n; ++i) {
          Scalar dx = x_values[i] - mean_x;
          Scalar dy = y_values[i] - mean_y;
          numerator += weights[i] * dx * dy;
          denominator += weights[i] * dx * dx;
        }
        Scalar slope = (denominator > FMCA_ZERO_TOLERANCE)
                           ? numerator / denominator
                           : dtilde_;
        Scalar intercept = mean_y - slope * mean_x;
        // R2
        Scalar ss_tot = 0.0;
        Scalar ss_res = 0.0;
        for (size_t i = 0; i < filtered_n; ++i) {
          Scalar y_pred = intercept + slope * x_values[i];
          ss_tot += weights[i] * (y_values[i] - mean_y) * (y_values[i] - mean_y);
          ss_res += weights[i] * (y_values[i] - y_pred) * (y_values[i] - y_pred);
        }
        Scalar r_squared = (ss_tot > FMCA_ZERO_TOLERANCE) ?
                           1.0 - (ss_res / ss_tot) : 1.0;

        results[leaf] = FitResult<Scalar>(std::min(dtilde_, slope), intercept, r_squared);
      }
    }
    return results;
  }

  //////////////////////////////////////////////////////////////////////////////

 private:
  LeafDataMap leaf_data_;
  Scalar dtilde_;
  Scalar small_threshold_;
};

}  // namespace FMCA
#endif  // FMCA_EDGEDETECTION_SLOPEFITTER_H

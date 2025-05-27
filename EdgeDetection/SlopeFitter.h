#ifndef FMCA_EDGEDETECTION_SLOPEFITTER_H
#define FMCA_EDGEDETECTION_SLOPEFITTER_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Macros.h"

namespace FMCA {
template <typename Scalar>
struct FitResult {
 private:
  Scalar slope;
  Scalar constant;

 public:
  FitResult(Scalar get_slope = 0, Scalar get_constant = 0)
      : slope(get_slope), constant(get_constant) {}

  Scalar get_slope() const { return slope; }
  Scalar get_constant() const { return constant; }
};

//////////////////////////////////////////////////////////////////////////////
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
  void init(const LeafDataMap& leaf_data, Scalar dtilde,
            Scalar small_threshold) {
    leaf_data_ = leaf_data;
    dtilde_ = dtilde > 0 ? dtilde : 1;
    small_threshold_ =
        small_threshold >= 0 ? small_threshold : FMCA_ZERO_TOLERANCE;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  ResultMap fitSlope() {
    ResultMap results;

    for (const auto& [leaf, dataPair] : leaf_data_) {
      auto coefficients = dataPair.first;
      auto diameters = dataPair.second;
      size_t n = coefficients.size();

      // First, filter out infinity values
      std::vector<Scalar> filtered_coefficients;
      std::vector<Scalar> filtered_diameters;

      for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(coefficients[i])) {
          filtered_coefficients.push_back(coefficients[i]);
          filtered_diameters.push_back(diameters[i]);
        }
      }

      // Use the filtered data for the rest of the calculations
      size_t filtered_n = filtered_coefficients.size();

      // Compute norm of filtered coefficients
      Scalar norm = std::sqrt(std::accumulate(
          filtered_coefficients.begin(), filtered_coefficients.end(), Scalar(0),
          [](Scalar sum, Scalar c) { return sum + c * c; }));

      // Edge case 1: Too few coefficients
      if (filtered_n < 2) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0);
      }

      // Edge case 2: Smooth decay, last two coefficients are near zero
      else if (std::abs(filtered_coefficients[filtered_n - 1] / norm) <
                   small_threshold_ &&
               std::abs(filtered_coefficients[filtered_n - 2] / norm) <
                   small_threshold_) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0);
      }

      else {
        // Solve Ax = b using QR decomposition
        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> A(filtered_n, 2);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(filtered_n);

        for (size_t i = 0; i < filtered_n; ++i) {
          A(i, 0) = 1.0;  // Constant term
          A(i, 1) = (filtered_diameters[i] == 0)
                        ? std::log(FMCA_ZERO_TOLERANCE)
                        : std::log(std::abs(
                              filtered_diameters[i]));  // Log of diameters
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

        // 2. Compute y = Q^T * b
        Eigen::Matrix<Scalar, 2, 1> y = Q.transpose() * b;

        // 3. Solve R x = y
        Eigen::Matrix<Scalar, 2, 1> x =
            R.template triangularView<Eigen::Upper>().solve(y);

        // x(0) = intercept, x(1) = slope
        results[leaf] = FitResult<Scalar>(std::min(dtilde_, x(1)), x(0));
      }
    }
    return results;
  }

  //////////////////////////////////////////////////////////////////////////////
  ResultMap fitSlopeRegression(bool weighted = false) {
    ResultMap results;

    for (const auto& [leaf, dataPair] : leaf_data_) {
      auto coefficients = dataPair.first;
      const auto& diameters = dataPair.second;
      size_t n = coefficients.size();

      // First, filter out infinity values
      std::vector<Scalar> filtered_coefficients;
      std::vector<Scalar> filtered_diameters;

      for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(coefficients[i])) {
          filtered_coefficients.push_back(coefficients[i]);
          filtered_diameters.push_back(diameters[i]);
        }
      }

      // Use the filtered data for the rest of the calculations
      size_t filtered_n = filtered_coefficients.size();

      // Compute norm of filtered coefficients
      Scalar norm = std::sqrt(std::accumulate(
          filtered_coefficients.begin(), filtered_coefficients.end(), Scalar(0),
          [](Scalar sum, Scalar c) { return sum + c * c; }));

      // Edge case 1: Too few coefficients
      if (filtered_n < 2) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0);
      }

      // Edge case 2: Smooth decay, last two coefficients are near zero
      else if (std::abs(filtered_coefficients[filtered_n - 1] / norm) <
                   small_threshold_ &&
               std::abs(filtered_coefficients[filtered_n - 2] / norm) <
                   small_threshold_) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0);
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

        // Calculate means
        Scalar mean_x =
            std::accumulate(x_values.begin(), x_values.end(), Scalar(0)) /
            filtered_n;
        Scalar mean_y =
            std::accumulate(y_values.begin(), y_values.end(), Scalar(0)) /
            filtered_n;

        // Set up weights if weighted regression is requested
        std::vector<Scalar> weights(filtered_n, Scalar(1.0));

        if (weighted) {
          // Calculate standard deviation of x_values
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

          // Normalize weights to sum to filtered_n
          Scalar sum_weights =
              std::accumulate(weights.begin(), weights.end(), Scalar(0));
          if (sum_weights > FMCA_ZERO_TOLERANCE) {
            Scalar scale = Scalar(filtered_n) / sum_weights;
            for (auto& w : weights) w *= scale;
          }
        }

        // Calculate slope and intercept
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

        results[leaf] = FitResult<Scalar>(std::min(dtilde_, slope), intercept);
      }
    }

    return results;
  }

  //////////////////////////////////////////////////////////////////////////////
  ResultMap fitSlopeRegressionLog2(bool weighted = false) {
    ResultMap results;

    for (const auto& [leaf, dataPair] : leaf_data_) {
      auto coefficients = dataPair.first;
      const auto& diameters = dataPair.second;
      size_t n = coefficients.size();

      // First, filter out infinity values
      std::vector<Scalar> filtered_coefficients;
      std::vector<Scalar> filtered_diameters;

      for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(coefficients[i])) {
          filtered_coefficients.push_back(coefficients[i]);
          filtered_diameters.push_back(diameters[i]);
        }
      }

      // Use the filtered data for the rest of the calculations
      size_t filtered_n = filtered_coefficients.size();

      // Compute norm of filtered coefficients
      Scalar norm = std::sqrt(std::accumulate(
          filtered_coefficients.begin(), filtered_coefficients.end(), Scalar(0),
          [](Scalar sum, Scalar c) { return sum + c * c; }));

      // Edge case 1: Too few coefficients
      if (filtered_n < 2) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0);
      }

      // Edge case 2: Smooth decay, last two coefficients are near zero
      else if (std::abs(filtered_coefficients[filtered_n - 1] / norm) <
                   small_threshold_ &&
               std::abs(filtered_coefficients[filtered_n - 2] / norm) <
                   small_threshold_) {
        results[leaf] = FitResult<Scalar>(dtilde_, 0);
      } else {
        std::vector<Scalar> x_values, y_values;
        x_values.reserve(filtered_n);
        y_values.reserve(filtered_n);

        for (size_t i = 0; i < filtered_n; ++i) {
          auto coef =
              std::max(std::abs(filtered_coefficients[i]), FMCA_ZERO_TOLERANCE);
          Scalar x_val = static_cast<Scalar>(i);
          x_values.emplace_back(-x_val);
          y_values.emplace_back(std::log2(coef));
        }

        // Calculate means
        Scalar mean_x =
            std::accumulate(x_values.begin(), x_values.end(), Scalar(0)) /
            filtered_n;
        Scalar mean_y =
            std::accumulate(y_values.begin(), y_values.end(), Scalar(0)) /
            filtered_n;

        // Set up weights if weighted regression is requested
        std::vector<Scalar> weights(filtered_n, Scalar(1.0));

        if (weighted) {
          // Calculate standard deviation of x_values
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

          // Normalize weights to sum to filtered_n
          Scalar sum_weights =
              std::accumulate(weights.begin(), weights.end(), Scalar(0));
          if (sum_weights > FMCA_ZERO_TOLERANCE) {
            Scalar scale = Scalar(filtered_n) / sum_weights;
            for (auto& w : weights) w *= scale;
          }
        }

        // Calculate slope and intercept
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

        results[leaf] = FitResult<Scalar>(std::min(dtilde_, slope), intercept);
      }
    }

    return results;
  }

  //////////////////////////////////////////////////////////////////////////////
  ResultMap fitSlopeRelative() {
    std::map<const TreeType*, std::vector<Scalar>> relative_result;

    for (const auto& [leaf, dataPair] : leaf_data_) {
      auto coefficients = dataPair.first;
      // const auto& diameters = dataPair.second;
      size_t n = coefficients.size();

      // Filter out NaN/infinite values
      std::vector<Scalar> filtered_coefficients;
      // std::vector<Scalar> filtered_diameters;

      for (size_t i = 0; i < n; ++i) {
        if (std::isfinite(coefficients[i])) {
          filtered_coefficients.push_back(coefficients[i]);
          // filtered_diameters.push_back(diameters[i]);
        }
      }

      size_t filtered_n = filtered_coefficients.size();

      if (filtered_n < 2) {
        continue;  // Need at least 2 points for slope calculation
      }

      relative_result[leaf].resize(filtered_n - 1);

      for (size_t i = 1; i < filtered_n; ++i) {
        auto coef0 = std::max(std::abs(filtered_coefficients[i - 1]),
                              FMCA_ZERO_TOLERANCE);
        auto coef1 =
            std::max(std::abs(filtered_coefficients[i]), FMCA_ZERO_TOLERANCE);
        Scalar x_0 = std::log(filtered_diameters[i - 1]);
        Scalar x_1 = std::log(filtered_diameters[i]);
        Scalar y_0 = std::log(coef0);
        Scalar y_1 = std::log(coef1);
        relative_result[leaf][i - 1] = (y_1 - y_0) / (x_1 - x_0);
        // (x_1 - x_0);
      }
    }
    return relative_result;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Getter
  Scalar dtilde() const { return dtilde_; }
  Scalar smallThreshold() const { return small_threshold_; }

 private:
  LeafDataMap leaf_data_;
  Scalar dtilde_;
  Scalar small_threshold_;
};

}  // namespace FMCA
#endif  // FMCA_EDGEDETECTION_SLOPEFITTER_H
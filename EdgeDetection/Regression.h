#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/IO.h"
#include "../FMCA/src/util/Macros.h"

using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;

using namespace FMCA;

// --------------------------------------------------------------------------------------------------
template <typename Scalar>
struct LinearRegressionResult {
 private:
  Scalar regression_slope;
  Scalar regression_intercept;

 public:
  LinearRegressionResult(Scalar slope = 0, Scalar intercept = 0)
      : regression_slope(slope), regression_intercept(intercept) {}

  Scalar slope() const { return regression_slope; }
  Scalar intercept() const { return regression_intercept; }

  //   // Mutator methods (optional, if you want to modify the values later)
  //   void setSlope(Scalar slope) { m_slope = slope; }
  //   void setIntercept(Scalar intercept) { m_intercept = intercept; }
};

// --------------------------------------------------------------------------------------------------
/**
 * Computes the slope and intercept of a weighted linear regression using the
 * given data points.
 *
 * The weights are computed as 1 / (x_i - meanX)^2, where meanX is the mean of
 * the x values. Points with x_i == meanX are skipped to avoid infinite weights.
 *
 * If all points get skipped, the function returns {0, 0}.
 *
 * @param x a vector of x values
 * @param y a vector of y values
 * @return a pair of slope and intercept of the weighted linear regression line
 */
template <typename Scalar>
LinearRegressionResult<Scalar> WeightedLinearRegression(
    const std::vector<Scalar>& x, const std::vector<Scalar>& y) {
  if (x.size() != y.size() || x.size() < 2) {
    return LinearRegressionResult<Scalar>(Scalar(0), Scalar(0));
  }
  // Compute mean
  Scalar sumX = 0;
  for (auto xi : x) sumX += xi;
  Scalar meanX = sumX / x.size();
  // Compute the sums needed for weighted linear regression
  //    slope = [Σ(w_i)*Σ(w_i*x_i*y_i) - Σ(w_i*x_i)*Σ(w_i*y_i)]
  //            / [Σ(w_i)*Σ(w_i*x_i^2) - (Σ(w_i*x_i))^2]
  //
  //    intercept = [Σ(w_i*y_i) - slope * Σ(w_i*x_i)] / Σ(w_i)
  //
  // where w_i = 1 / (x_i - meanX)^2, skipping points where x_i == meanX
  Scalar sumW = 0, sumWX = 0, sumWY = 0, sumWXY = 0, sumWX2 = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    Scalar dist = x[i] - meanX;
    if (std::abs(dist) < std::numeric_limits<Scalar>::epsilon()) {
      continue;
    }
    Scalar w = Scalar(1) / (dist * dist);
    sumW += w;
    sumWX += w * x[i];
    sumWY += w * y[i];
    sumWXY += w * x[i] * y[i];
    sumWX2 += w * x[i] * x[i];
  }
  // If all points got skipped, then we have an horizontal line
  if (sumW <= 0) {
    return LinearRegressionResult<Scalar>(Scalar(0),
                                          y.empty() ? Scalar(0) : y[0]);
  }
  Scalar denominator = (sumW * sumWX2 - sumWX * sumWX);
  if (std::abs(denominator) < FMCA_ZERO_TOLERANCE) {
    return LinearRegressionResult<Scalar>(Scalar(0), Scalar(0));
  }
  Scalar slope = (sumW * sumWXY - sumWX * sumWY) / denominator;
  Scalar intercept = (sumWY - slope * sumWX) / sumW;
  return LinearRegressionResult<Scalar>(slope, intercept);
}

// --------------------------------------------------------------------------------------------------
/**
 * Computes the linear regression of a set of points.
 *
 * Given two vectors of equal length `x` and `y`, computes the linear
 * regression of the points `(x[i], y[i])`, returning a `LinearRegressionResult`
 * containing the slope and intercept of the regression line.
 *
 * The slope is computed using the formula:
 *    slope = (n * Σ(xy) - Σ(x) * Σ(y)) / (n * Σ(x^2) - (Σ(x))^2)
 *
 * The intercept is computed using the formula:
 *    intercept = (Σ(y) - slope * Σ(x)) / n
 *
 * If the vectors are empty, or if all `x` values are equal, then the slope
 * is set to 0 and the intercept is set to the average of the `y` values (or 0
 * if `y` is also empty).
 *
 * @param x the vector of x values
 * @param y the vector of y values
 * @return a `LinearRegressionResult` containing the slope and intercept of the
 * regression line
 */
template <typename Scalar>
LinearRegressionResult<Scalar> LinearRegression(const std::vector<Scalar>& x,
                                                const std::vector<Scalar>& y) {
  if (x.size() != y.size() || x.size() < 2) {
    return LinearRegressionResult<Scalar>(Scalar(0), Scalar(0));
  }

  // Compute sums needed for linear regression
  Scalar sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
  }

  // Compute slope and intercept
  Scalar denominator = (n * sumX2 - sumX * sumX);
  if (std::abs(denominator) < std::numeric_limits<Scalar>::epsilon()) {
    // Degenerate case: all x values are the same
    return LinearRegressionResult<Scalar>(Scalar(0),
                                          y.empty() ? Scalar(0) : y[0]);
  }

  Scalar slope = (n * sumXY - sumX * sumY) / denominator;
  Scalar intercept = (sumY - slope * sumX) / n;

  return LinearRegressionResult<Scalar>(slope, intercept);
}

// // --------------------------------------------------------------------------------------------------
// /**
//  * Computes the linear regression slope for each leaf in the tree, given the
//  * coefficients of each leaf.
//  *
//  * The slope is computed using linear regression on the log2 of the coefficients
//  * of each leaf. The slope is then saved to a file, along with the bounding box
//  * of the leaf.
//  *
//  * @param leafCoefficients A map of leaves to their coefficients
//  * @param dtilde The threshold below which the slope is set to -dtilde
//  * @param outputFilename The name of the output file
//  * @return A map of leaves to their slopes
//  */
// template <typename TreeType, typename Scalar>
// std::map<const TreeType*, Scalar> computeLinearRegressionSlope(
//     const std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients,
//     const Scalar& dtilde, const std::string& outputFilename) {
//   std::map<const TreeType*, Scalar> slopes;

//   // Open a file to save the results
//   std::ofstream outputFile(outputFilename);
//   outputFile << "x_min x_max Slope\n";  // CSV header

//   for (const auto& [leaf, coefficients] : leafCoefficients) {
//     size_t n = coefficients.size();
//     std::vector<Scalar> x, y;

//     for (size_t i = 1; i < n; ++i) {
//       x.push_back(static_cast<Scalar>(i));      // Levels 1, 2, ..., n-2
//       y.push_back(std::log2(coefficients[i]));  // Log2 of the coefficients
//     }

//     // Compute the slope using linear regression
//     Scalar sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
//     size_t m = x.size();
//     for (size_t i = 0; i < m; ++i) {
//       sumX += x[i];
//       sumY += y[i];
//       sumXY += x[i] * y[i];
//       sumX2 += x[i] * x[i];
//     }

//     Scalar slope = (m * sumXY - sumX * sumY) / (m * sumX2 - sumX * sumX);
//     slopes[leaf] = slope;

//     // Save the leaf pointer (as an address) and slope to the file
//     outputFile << leaf->bb()(0) << " " << leaf->bb()(1) << " " << std::fixed
//                << std::setprecision(6) << slope << "\n";
//   }

//   outputFile.close();
//   return slopes;
// }

template <typename TreeType>
std::map<const TreeType*, Scalar> computeLinearRegressionSlope(
    const std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients,
    const Scalar& dtilde) {
  std::map<const TreeType*, Scalar> slopes;

  for (const auto& [leaf, coefficients] : leafCoefficients) {
    size_t n = coefficients.size();

    // Exclude the first and last coefficients
    std::vector<Scalar> x, y;
    int counter_small_coefficients = 0;
    // Scalar log10_sum_small_coefficients = 0.0;
    for (size_t i = 1; i < n; ++i) {
      x.push_back(static_cast<Scalar>(i));      // Levels 1, 2, ..., n-2
      y.push_back(std::log2(coefficients[i]));  // Log2 of the coefficients
      if (coefficients[i] < 1e-6) {
        counter_small_coefficients++;
      }
    }
    // if counter is greather that half of the coefficients, set the slope to -dtilde
    if (counter_small_coefficients > n / 3) {
      slopes[leaf] = -dtilde;
    } else {
      // Compute the slope using linear regression
      Scalar sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
      size_t m = x.size();
      for (size_t i = 0; i < m; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
      }

      Scalar slope = (m * sumXY - sumX * sumY) / (m * sumX2 - sumX * sumX);
      slopes[leaf] = slope;
    }
  }

  return slopes;
}

// --------------------------------------------------------------------------------------------------
/**
 * Computes the magnitude ratios for each leaf in the tree, given the
 * coefficients of each leaf.
 *
 * The method works as follows:
 * 1. For each leaf, check if the magnitude ratios (i.e. |c_i/c_{i-1}|) are
 * below the given threshold. If they are, then we suspect a "fast drop" to
 * zero. In this case, we do a Weighted Linear Regression on the subsequent
 * points to confirm. If the intercept of the regression is below a certain
 * threshold (`smallThreshold`), then we conclude that the coefficients have
 * effectively dropped to 0.
 * 2. If no magnitude ratio exceeded the threshold, then we compute the Weighted
 * Linear Regression on the entire set of coefficients. The slope of this
 * regression is the relative slope for the leaf.
 *
 * The results are stored in the output file in CSV format, with columns for the
 * minimum and maximum x values of the leaf, and the relative slope.
 *
 * @param leafCoefficients The map of leaf coefficients.
 * @param dtilde The threshold for the magnitude ratio check.
 * @param threshold The threshold for the intercept of the linear regression.
 * @param outputFilename The name of the output file.
 * @return A map of leaves to their relative slopes.
 */
template <typename TreeType, typename Scalar>
std::map<const TreeType*, Scalar> computeMagnitudeRatios(
    const std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients,
    const Scalar& dtilde, const Scalar& threshold,
    const std::string& outputFilename) {
  std::map<const TreeType*, Scalar> slopes;

  // Open file for output
  std::ofstream outputFile(outputFilename);
  outputFile << "x_min x_max MagnitudeRatio\n";  // CSV header

  for (const auto& [leaf, coefficients] : leafCoefficients) {
    size_t n = coefficients.size();

    if (n < 2) {
      // Not enough coefficients to compute ratios, assign default slope
      slopes[leaf] = -dtilde;
      outputFile << leaf->bb()(0) << " " << leaf->bb()(1) << " " << -dtilde
                 << "\n";
      continue;
    }

    Scalar slope = 0;
    bool thresholdViolated = false;
    Scalar sumRatios = 0;
    size_t validRatios = 0;

    for (size_t i = 1; i < n; ++i) {
      Scalar ratio = std::abs(coefficients[i] / coefficients[i - 1]);

      if (ratio < threshold) {
        slope = -dtilde;
        thresholdViolated = true;
        break;
      }

      sumRatios += ratio;
      ++validRatios;
    }

    if (!thresholdViolated) {
      // Compute the slope using linear regression
      std::vector<Scalar> x, y;
      for (size_t i = 1; i < n; ++i) {
        x.push_back(static_cast<Scalar>(i));      // Levels: 1, 2, ..., n-1
        y.push_back(std::log2(coefficients[i]));  // Log2 of coefficients
      }

      // Linear regression calculation
      Scalar sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
      size_t m = x.size();
      for (size_t i = 0; i < m; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
      }

      slope = (m * sumXY - sumX * sumY) / (m * sumX2 - sumX * sumX);
    }

    slopes[leaf] = slope;
    outputFile << leaf->bb()(0) << " " << leaf->bb()(1) << " " << std::fixed
               << std::setprecision(6) << slope << "\n";
  }

  outputFile.close();
  return slopes;
}

// --------------------------------------------------------------------------------------------------
/**
 * Computes the relative slopes for a given set of leaf coefficients.
 *
 * The method works as follows:
 * 1. For each leaf, check if the local slopes (i.e. |log2(c_i) -
 * log2(c_{i-1})|) exceed the given threshold `dtilde`. If a local slope exceeds
 * `dtilde`, then we suspect a "fast drop" to zero. In this case, we do a
 * Weighted Linear Regression on the subsequent points to confirm. If the
 * intercept of the regression is below a certain threshold (`smallThreshold`),
 * then we conclude that the coefficients have effectively dropped to 0.
 * 2. If no local slope exceeded `dtilde`, then we compute the Weighted Linear
 * Regression on the entire set of coefficients. The slope of this regression is
 * the relative slope for the leaf.
 *
 * The results are stored in the output file in CSV format, with columns for the
 * minimum and maximum x values of the leaf, and the relative slope.
 *
 * @param leafCoefficients The map of leaf coefficients.
 * @param dtilde The threshold for the local slope check.
 * @param outputFilename The name of the output file.
 * @return A map of leaves to their relative slopes.
 */
template <typename TreeType, typename Scalar>
std::map<const TreeType*, Scalar> computeRelativeSlopes1D(
    const std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients,
    const Scalar& dtilde, const Scalar& interceptThreshold,
    const std::string& outputFilename) {
  std::map<const TreeType*, Scalar> slopes;

  // Open file for output
  std::ofstream outputFile(outputFilename);
  outputFile << "x_min x_max RelativeSlopes\n";  // CSV header

  for (const auto& [leaf, coefficients] : leafCoefficients) {
    size_t n = coefficients.size();

    if (n < 2) {
      // Not enough coefficients to compute ratios, assign default slope
      slopes[leaf] = -dtilde;
      outputFile << leaf->bb()(0) << " " << leaf->bb()(1) << " " << -dtilde
                 << "\n";
      continue;
    }

    if (std::all_of(
            coefficients.begin(), coefficients.end(),
            [&](const Scalar& coeff) { return coeff < FMCA_ZERO_TOLERANCE; })) {
      slopes[leaf] = -dtilde;
      outputFile << leaf->bb()(0) << " " << leaf->bb()(1) << " " << -dtilde
                 << "\n";
      continue;
    }

    Scalar slope = 0;
    bool thresholdViolated = false;

    // Check local slopes to see if they exceed dtilde
    for (size_t i = 1; i < n; ++i) {
      if (coefficients[i] == 0 ||
          coefficients[i - 1] == 0) {
        continue;
      }
      Scalar localSlope =
          std::abs(std::log2(coefficients[i]) - std::log2(coefficients[i - 1]));

      if (localSlope >= dtilde) {
        // We suspect a "fast drop" to zero.
        // Gather next points in log2 space (skipping the one that triggered the
        // violation)
        std::vector<Scalar> xNext, yNext;
        for (size_t j = i; j < n; ++j) {
          xNext.push_back(static_cast<Scalar>(j));
          yNext.push_back(std::log2(coefficients[j]));
        }

        auto regression_result = WeightedLinearRegression<Scalar>(xNext, yNext);
        // If 2^(interceptNext) < interceptThreshold => coefficients effectively
        // 0
        if (std::pow(Scalar(2), regression_result.intercept()) <
            interceptThreshold) {
          // Conclude dropped to 0: slope = -dtilde
          slope = -dtilde;
          thresholdViolated = true;
        }
        break;
      }
    }

    // If threshold not violated, compute Weighted Linear Regression
    //    on the entire set [from i=1 to n-1].
    if (!thresholdViolated) {
      std::vector<Scalar> xAll, yAll;
      for (size_t i = 0; i < n; ++i) {
        xAll.push_back(static_cast<Scalar>(i));
        yAll.push_back(std::log2(coefficients[i]));
      }
      auto regression_result = WeightedLinearRegression<Scalar>(xAll, yAll);
      slope = regression_result.slope();
    }

    slopes[leaf] = slope;
    outputFile << leaf->bb()(0) << " " << leaf->bb()(1) << " " << std::fixed
               << std::setprecision(6) << slope << "\n";
  }

  outputFile.close();
  return slopes;
}



// --------------------------------------------------------------------------------------------------
/**
 * Computes the relative slopes for a given set of leaf coefficients in 2D.
 *
 * The method works as follows:
 * 1. For each leaf, check if the local slopes (i.e. |log2(c_i) -
 * log2(c_{i-1})|) exceed the given threshold `dtilde`. If a local slope exceeds
 * `dtilde`, then we suspect a "fast drop" to zero. In this case, we do a
 * Weighted Linear Regression on the subsequent points to confirm. If the
 * intercept of the regression is below a certain threshold (`interceptThreshold`),
 * then we conclude that the coefficients have effectively dropped to 0.
 * 2. If no local slope exceeded `dtilde`, then we compute the Weighted Linear
 * Regression on the entire set of coefficients. The slope of this regression is
 * the relative slope for the leaf.
 *
 * The results are stored in the output file in CSV format, with columns for the
 * minimum and maximum x and y values of the leaf, and the relative slope.
 *
 * @param leafCoefficients The map of leaf coefficients.
 * @param dtilde The threshold for the local slope check.
 * @param interceptThreshold The threshold below which the intercept of the
 * Weighted Linear Regression is considered to be effectively 0.
 * @param outputFilename The name of the output file.
 * @return A map of leaves to their relative slopes.
 */
template <typename TreeType, typename Scalar>
std::map<const TreeType*, Scalar> computeRelativeSlopes2D(
    const std::map<const TreeType*, std::vector<Scalar>>& leafCoefficients,
    const Scalar& dtilde, const Scalar& interceptThreshold,
    const std::string& outputFilename) {
  
  std::map<const TreeType*, Scalar> slopes;

  // Open file for output
  std::ofstream outputFile(outputFilename);
  outputFile << "x_min x_max y_min y_max RelativeSlopes\n";  // CSV header

  for (const auto& [leaf, coefficients] : leafCoefficients) {
    size_t n = coefficients.size();

    if (n < 2) {
      // Not enough coefficients to compute ratios, assign default slope
      slopes[leaf] = -dtilde;
      outputFile << leaf->bb()(0,0) << " " << leaf->bb()(0,1) << " "
                 << leaf->bb()(1,0) << " " << leaf->bb()(1,1) << " " << -dtilde
                 << "\n";
      continue;
    }

    if (std::all_of(
            coefficients.begin(), coefficients.end(),
            [&](const Scalar& coeff) { return coeff < FMCA_ZERO_TOLERANCE; })) {
      slopes[leaf] = -dtilde;
      outputFile << leaf->bb()(0,0) << " " << leaf->bb()(0,1) << " "
                 << leaf->bb()(1,0) << " " << leaf->bb()(1,1) << " " << -dtilde
                 << "\n";
      continue;
    }

    Scalar slope = 0.;
    bool thresholdViolated = false;

    // Check local slopes to see if they exceed dtilde
    for (size_t i = 1; i < n; ++i) {
      Scalar coeffCurrent = coefficients[i] == 0 ? FMCA_ZERO_TOLERANCE : coefficients[i];
      Scalar coeffPrev = coefficients[i - 1] == 0 ? FMCA_ZERO_TOLERANCE : coefficients[i - 1];

      Scalar localSlope =
          std::abs(std::log2(coefficients[i]) - std::log2(coefficients[i - 1]));

      if (localSlope >= dtilde) {
        // We suspect a "fast drop" to zero.
        std::vector<Scalar> xNext, yNext;
        for (size_t j = i; j < n; ++j) {
          Scalar coeffAdjusted = coefficients[j] == 0 ? FMCA_ZERO_TOLERANCE : coefficients[j];
          xNext.push_back(static_cast<Scalar>(j));
          yNext.push_back(std::log2(coeffAdjusted));
        }

        auto regression_result = WeightedLinearRegression<Scalar>(xNext, yNext);
        
        if (std::pow(Scalar(2), regression_result.intercept()) <
            interceptThreshold) {
          // Conclude dropped to 0: slope = -dtilde
          slope = -dtilde;
          thresholdViolated = true;
        }
        break;
      }
    }

    // If threshold not violated, compute Weighted Linear Regression
    if (!thresholdViolated) {
      std::vector<Scalar> xAll, yAll;
      for (size_t i = 0; i < n; ++i) {
        Scalar coeffAdjusted = coefficients[i] == 0 ? FMCA_ZERO_TOLERANCE : coefficients[i];
        xAll.push_back(static_cast<Scalar>(i));
        yAll.push_back(std::log2(coeffAdjusted));
      }
      auto regression_result = WeightedLinearRegression<Scalar>(xAll, yAll);
      slope = regression_result.slope();
    }

    slopes[leaf] = slope;
    outputFile << leaf->bb()(0,0) << " " << leaf->bb()(0,1) << " "
               << leaf->bb()(1,0) << " " << leaf->bb()(1,1) << " "
               << std::fixed << std::setprecision(6) << slope << "\n";
  }

  outputFile.close();
  return slopes;
}



// --------------------------------------------------------------------------------------------------
template <typename TreeType, typename Scalar>
std::map<const TreeType*, Scalar> computeRelativeSlopesDiameters(
    const std::map<const TreeType*, std::pair<std::vector<Scalar>, std::vector<Scalar>>>& leafData,
    const Scalar& dtilde, 
    const Scalar& interceptThreshold){

    std::map<const TreeType*, Scalar> slopes;
    // Helper lambda to do log_{diam}(coeff), with some minimal checks
    auto logDiam = [&](Scalar coeff, Scalar diam) {
        Scalar c = (coeff < FMCA_ZERO_TOLERANCE ) ? FMCA_ZERO_TOLERANCE : coeff;
        Scalar d = (diam < FMCA_ZERO_TOLERANCE)  ? FMCA_ZERO_TOLERANCE : diam;
        return std::log2(c) / std::log2(d);
    };

    for (const auto& [leaf, dataPair] : leafData) {
        const auto& coefficients = dataPair.first;
        const auto& diameters    = dataPair.second;
        size_t n = coefficients.size();
        // Extreme case, just 1 coeff --> slope = -dtilde
        if (n < 2) {
            slopes[leaf] = -dtilde;
            continue;
        }
         //  Extreme case, all the coeffs are closed to 0 --> slope = -dtilde
        if (std::all_of(coefficients.begin(), coefficients.end(),
                        [&](Scalar c){return c < FMCA_ZERO_TOLERANCE;})) {
            slopes[leaf] = -dtilde;
            continue;
        }

        Scalar slope = 0.;
        bool thresholdViolated = false;
        // 1) Check local slopes for "fast drop"
        for (size_t i = 1; i < n; ++i) {
            Scalar localSlope = std::abs(logDiam(coefficients[i],   diameters[i]) -
                                         logDiam(coefficients[i-1], diameters[i-1]));

            if (localSlope >= dtilde) {
                // We suspect a fast drop => do WeightedRegression with the remaining coeffs to observe the intercept
                std::vector<Scalar> xNext, yNext;
                xNext.reserve(n - i);
                yNext.reserve(n - i);
                for (size_t j = i; j < n; ++j) {
                    xNext.push_back(static_cast<Scalar>(j));
                    yNext.push_back(logDiam(coefficients[j], diameters[j]));
                }
                auto regResult = WeightedLinearRegression<Scalar>(xNext, yNext);
                if (std::pow(2, regResult.intercept()) < interceptThreshold) {
                    // Conclude it dropped to ~0
                    slope = -dtilde;
                    thresholdViolated = true;
                }
                break;  // break from localSlope loop
            }
        }
        // 2) If no threshold violation, compute WeightedRegression over all
        if (!thresholdViolated) {
            std::vector<Scalar> xAll, yAll;
            xAll.reserve(n);
            yAll.reserve(n);

            for (size_t i = 0; i < n; ++i) {
                xAll.push_back(static_cast<Scalar>(i));
                yAll.push_back(logDiam(coefficients[i], diameters[i]));
            }
            auto regResult = WeightedLinearRegression<Scalar>(xAll, yAll);
            slope = regResult.slope();
        }
    }
    return slopes;
}


// --------------------------------------------------------------------------------------------------
template <typename TreeType, typename Scalar>
std::map<const TreeType*, Scalar> computeSSparsity(
    const std::map<const TreeType*, std::pair<std::vector<Scalar>, std::vector<Scalar>>>& leafData,
    const Scalar& dtilde, 
    const Scalar& smallThreshold){
    std::map<const TreeType*, Scalar> slopes;
    // Helper lambda to compute log_{diam}(coeff)
    auto logDiam = [&](Scalar coeff, Scalar diam) {
        Scalar c = (coeff == 0) ? FMCA_ZERO_TOLERANCE : coeff;
        Scalar d = (diam == 0)  ? FMCA_ZERO_TOLERANCE : diam;
        return std::log2(c) / std::log2(d);
    };

    for (const auto& [leaf, dataPair] : leafData) {
        auto coefficients = dataPair.first;
        const auto& diameters = dataPair.second;
        size_t n = coefficients.size();
        Scalar accum = 0.;
        for (int i = 0; i < n; ++i) {
            accum += coefficients[i] * coefficients[i];
        }
        Scalar norm = sqrt(accum);
        // if (norm != 0) {
        //   for (auto& coeff : coefficients) {
        //       coeff /= norm;
        //   }
        // }

        // Extreme case, just 1 coeff --> slope = -dtilde
        if (n < 2) {
            slopes[leaf] = -dtilde;
            continue;
        }
        //  Extreme case, all the coeffs are closed to 0 --> slope = -dtilde
        if (std::all_of(coefficients.begin(), coefficients.end(),
                        [&](Scalar c){return c < FMCA_ZERO_TOLERANCE;})) {
            slopes[leaf] = -dtilde;
            continue;
        }

        Scalar slope = 0.;
        bool thresholdViolated = false;
        // Smooth decay --> if the last 2 coeffs are 0, then slope = -dtilde
        if (n >= 2 && 
            coefficients[n - 1] / norm < smallThreshold && coefficients[n - 2] / norm < smallThreshold) { // && coefficients[n - 2] < smallThreshold
            slopes[leaf] = -dtilde;
            continue;
        }

        // In all the other cases, compute the slope of the weighted regression
        std::vector<Scalar> xAll, yAll;
        xAll.reserve(n);
        yAll.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            xAll.push_back(static_cast<Scalar>(i));
            yAll.push_back(logDiam(coefficients[i], diameters[i]));
        }
        auto regResult = LinearRegression<Scalar>(xAll, yAll);
        slope = regResult.slope();
        slopes[leaf] = slope;
    }
    return slopes;
}


// --------------------------------------------------------------------------------------------------
template <typename TreeType>
Vector computeMallatModulusMaxima(const TreeType& st, const Scalar& level,
                                  const Vector& tdata) {
  // Create a vector to hold coefficients corresponding to points
  Vector Coeffs = Vector::Zero(tdata.size());

  for (const auto& node : st) {
    if (node.level() == level) {
      if (node.nsamplets() > 0 && node.start_index() >= 0 &&
          node.start_index() + node.nsamplets() <= tdata.size()) {
        // Extract segment of coefficients
        auto segment = tdata.segment(node.start_index(), node.nsamplets());

        // Access node indices
        const auto& indices = node.indices();
        for (Index i = 0; i < node.block_size(); ++i) {
          Coeffs[indices[i]] = std::abs(segment[i]);
        }
      }
    }
  }
  return Coeffs;
}

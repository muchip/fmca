#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

#include "../FMCA/H2Matrix"
#include "../FMCA/Samplets"
#include "../FMCA/src/util/Macros.h"

using namespace FMCA;

// --------------------------------------------------------------------------------------------------
template <typename Scalar>
struct FitResult {
 private:
  Scalar fit_dtilde;
  Scalar fit_constant;

 public:
  FitResult(Scalar get_dtilde = 0, Scalar get_constant = 0)
      : fit_dtilde(get_dtilde), fit_constant(get_constant) {}

  Scalar get_dtilde() const { return fit_dtilde; }
  Scalar get_constant() const { return fit_constant; }
};

// --------------------------------------------------------------------------------------------------
template <typename TreeType>
std::map<const TreeType*, FitResult<Scalar>> FitDtilde(
    const std::map<const TreeType*, std::pair<std::vector<Scalar>,
                                              std::vector<Scalar>>>& leafData,
    const Scalar& dtilde, const Scalar& smallThreshold) {
  std::map<const TreeType*, FitResult<Scalar>> results;

  for (const auto& [leaf, dataPair] : leafData) {
    auto coefficients = dataPair.first;
    const auto& diameters = dataPair.second;
    size_t n = coefficients.size();

    // Compute norm of coefficients
    Scalar norm = std::sqrt(
        std::accumulate(coefficients.begin(), coefficients.end(), Scalar(0),
                        [](Scalar sum, Scalar c) { return sum + c * c; }));

    // Edge case 1: Too few coefficients
    if (n < 2) {
      results[leaf] = FitResult<Scalar>(dtilde, 0);
      continue;
    }

    // Edge case 2: All coefficients are near zero
    if (std::all_of(coefficients.begin(), coefficients.end(), [&](Scalar c) {
          return std::abs(c) < FMCA_ZERO_TOLERANCE;
        })) {
      results[leaf] = FitResult<Scalar>(dtilde, 0);
      continue;
    }

    // Edge case 3: Smooth decay, last two coefficients are near zero
    if (n >= 2 && std::abs(coefficients[n - 1] / norm) < smallThreshold && std::abs(coefficients[n - 2] / norm) < smallThreshold)
    {
      results[leaf] = FitResult<Scalar>(dtilde, 0);
      continue;
    }

    // Solve Ax = b using QR decomposition
    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> A(n, 2);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);

    for (size_t i = 0; i < n; ++i) {
      A(i, 0) = 1.0;                     // Constant term
      A(i, 1) = std::log(diameters[i]);  // Log of diameters
      b(i) = (coefficients[i] == 0) ? FMCA_ZERO_TOLERANCE
                                    : std::log(std::abs(coefficients[i]));
    }

    // Compute the least squares solution using the QR decomposition
    Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 2>> qr(A);

    // Extract the "thin" Q by multiplying by Identity of shape Nx2
    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> Q =
        qr.householderQ() *
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
            A.rows(), 2);

    // Extract the top-left 2x2 block of R
    Eigen::Matrix<Scalar, 2, 2> R =
        qr.matrixQR()
            .topLeftCorner(2, 2)
            .template triangularView<Eigen::Upper>();

    // 2. Compute y = Q^T * b
    Eigen::Matrix<Scalar, 2, 1> y = Q.transpose() * b;

    // 3. Solve R x = y (since R is 2x2 upper-triangular)
    Eigen::Matrix<Scalar, 2, 1> x =
        R.template triangularView<Eigen::Upper>().solve(y);

    // x(0) = intercept (b), x(1) = slope (a)
    results[leaf] = FitResult<Scalar>(x(1), x(0));
  }

  return results;
}
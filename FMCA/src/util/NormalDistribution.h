// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_UTIL_NORMALDISTRIBUTION_H_
#define FMCA_UTIL_NORMALDISTRIBUTION_H_
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>

#include <Eigen/Dense>

namespace FMCA {
class NormalDistribution {
public:
  NormalDistribution(Scalar mu, Scalar sigma)
      : _normalDist(std::normal_distribution<Scalar>(mu, sigma)), _mu(mu),
        _sigma(sigma) {
    _mt64.seed(time(NULL));
  };
  NormalDistribution(Scalar mu, Scalar sigma, size_t seed)
      : _normalDist(std::normal_distribution<Scalar>(mu, sigma)), _mu(mu),
        _sigma(sigma) {
    _mt64.seed(seed);
  };
  /*
   *  plot histogram of normal distribution
   */
  void visDist(int intervals, int samples) {
    Scalar h = 6. * _sigma / intervals;
    Scalar rand_number;
    Eigen::VectorXd values;

    values.resize(intervals);
    values.setZero();
    for (auto i = 0; i < samples; ++i) {
      rand_number = _normalDist(_mt64);
      for (auto j = 0; j < intervals; ++j)
        if ((rand_number >= h * j - 3. * _sigma + _mu) &&
            (rand_number < h * (j + 1) - 3. * _sigma + _mu)) {
          ++values(j);
          break;
        }
    }

    std::cout << "normal_distribution (" << _mu << "," << _sigma
              << "):" << std::endl;

    for (auto i = 0; i < intervals; ++i) {
      std::cout << std::setprecision(2) << std::setw(10)
                << h * (i + 0.5) - 3. * _sigma + _mu << "|";
      std::cout << std::string(
                       std::round(20 * values(i) * _sigma / h / samples), '*')
                << std::endl;
    }
  }

  /*
   *  get Matrix of normally distributed random variables
   */
  const Matrix &get_randMat(Index m, Index n) {
    _randMat.resize(m, n);
    for (auto i = 0; i < m; ++i)
      for (auto j = 0; j < n; ++j)
        _randMat(i, j) = _normalDist(_mt64);

    return (const Matrix &)_randMat;
    // return (const Eigen::VectorXd &)_randMat;
  }

private:
  std::mt19937_64 _mt64;
  std::normal_distribution<Scalar> _normalDist;
  Scalar _mu;
  Scalar _sigma;
  Matrix _randMat;
};
} // namespace FMCA
#endif

// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_UTIL_NORMALDISTRIBUTION_H_
#define FMCA_UTIL_NORMALDISTRIBUTION_H_

#include <iomanip>
#include <iostream>
#include <random>

#include "Macros.h"

namespace FMCA {
class NormalDistribution {
 public:
  NormalDistribution(Scalar mu, Scalar sigma)
      : _normalDist(std::normal_distribution<Scalar>(mu, sigma)),
        _mu(mu),
        _sigma(sigma) {
    _mt64.seed(time(NULL));
  };
  NormalDistribution(Scalar mu, Scalar sigma, size_t seed)
      : _normalDist(std::normal_distribution<Scalar>(mu, sigma)),
        _mu(mu),
        _sigma(sigma) {
    _mt64.seed(seed);
  };
  /*
   *  plot histogram of normal distribution
   */
  void visDist(Index Indexervals, Index samples) {
    Scalar h = 6. * _sigma / Indexervals;
    Scalar rand_number;
    Vector values;

    values.resize(Indexervals);
    values.setZero();
    for (Index i = 0; i < samples; ++i) {
      rand_number = _normalDist(_mt64);
      for (Index j = 0; j < Indexervals; ++j)
        if ((rand_number >= h * j - 3. * _sigma + _mu) &&
            (rand_number < h * (j + 1) - 3. * _sigma + _mu)) {
          ++values(j);
          break;
        }
    }

    std::cout << "normal_distribution (" << _mu << "," << _sigma
              << "):" << std::endl;

    for (Index i = 0; i < Indexervals; ++i) {
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
  Matrix randN(Index m, Index n) {
    Matrix retval(m, n);
    for (Index i = 0; i < m; ++i)
      for (Index j = 0; j < n; ++j) retval(i, j) = _normalDist(_mt64);
    return retval;
  }

 private:
  std::mt19937_64 _mt64;
  std::normal_distribution<Scalar> _normalDist;
  Scalar _mu;
  Scalar _sigma;
};
}  // namespace FMCA
#endif

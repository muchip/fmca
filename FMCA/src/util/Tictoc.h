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
#ifndef FMCA_UTIL_TICTOC__
#define FMCA_UTIL_TICTOC__

#include <chrono>
#include <iostream>
#include <string>

#include "Macros.h"

namespace FMCA {
class Tictoc {
 public:
  void tic(void) { start_ = Clock::now(); }
  Scalar toc(void) {
    const std::chrono::duration<Scalar> dtime = Clock::now() - start_;
    return dtime.count();
  }
  Scalar toc(const std::string &message) {
    const Scalar dtime = toc();
    std::cout << message << " " << dtime << "sec.\n";
    return dtime;
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};
}  // namespace FMCA
#endif

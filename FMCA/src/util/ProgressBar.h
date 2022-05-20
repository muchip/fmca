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
#ifndef FMCA_UTIL_PROGRESSBAR_H_
#define FMCA_UTIL_PROGRESSBAR_H_

#include "Macros.h"

namespace FMCA {

class ProgressBar {
public:
  ProgressBar() : currStep_(0), numSteps_(0), displayNext_(1) {}
  ProgressBar(Index numSteps)
      : currStep_(0), numSteps_(numSteps), displayNext_(1) {}
  void reset(Index numSteps) {
    numSteps_ = numSteps;
    currStep_ = 0;
    displayNext_ = 1;
    percent_ = 0;
    return;
  }
  void next() {
    if (currStep_ < numSteps_)
      ++currStep_;
    percent_ = (100 * currStep_) / numSteps_;
    if (percent_ > 100)
      return;
    if (percent_ >= displayNext_) {
      std::cout << "\r"
                << "[" << std::string(percent_ / 4, '|')
                << std::string(100 / 4 - percent_ / 4, ' ') << "]";
      std::cout << " (" << percent_ << "%)" << std::flush;
      displayNext_ += 1;
    }
    return;
  }

private:
  Index currStep_;
  Index numSteps_;
  Index displayNext_;
  Index percent_;
};
} // namespace FMCA

#endif

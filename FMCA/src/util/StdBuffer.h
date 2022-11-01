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
#ifndef FMCA_UTIL_STDBUFFER_H_
#define FMCA_UTIL_STDBUFFER_H_

#include <cstdio>

namespace FMCA {
class StdoutBuffer {
 public:
  // the collector string is used for collecting the output to stdout
  StdoutBuffer(const std::string &collector)
      : fp_(std::fopen(collector.c_str(), "w")) {
    assert(fp_ && "could not open output bypass");
    std::swap(stdout, fp_);  // swap stdout and the temp file
  }
  ~StdoutBuffer() {
    std::swap(stdout, fp_);  // swap back
    std::fclose(fp_);
  }

 private:
  std::FILE *fp_;
};
}  // namespace FMCA
#endif

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
#ifndef FMCA_UTIL_MLMEMORYBUFFER_H_
#define FMCA_UTIL_MLMEMORYBUFFER_H_

#include <vector>

namespace FMCA {

template <typename T>
class MLMemoryBuffer {
 public:
  MLMemoryBuffer() {}
  MLMemoryBuffer(size_t levels) { init(levels); }

  void init(size_t levels) {
    capacity_.clear();
    size_.clear();
    buffer_.clear();
    capacity_.resize(levels);
    size_.resize(levels);
    buffer_.resize(levels);
  }

  void increaseCapacity(size_t level, size_t size) {
    capacity_[level] += size;
    return;
  }

  void alloc() {
    for (size_t i = 0; i < capacity_.size(); ++i) {
      buffer_[i].resize(capacity_[i]);
      size_[i] = 0;
    }
  }

  std::vector<T> &buffer(size_t level) { return buffer_[level]; }

  size_t &size(size_t level) { return size_[level]; }

  const size_t &capacity(size_t level) { return capacity_[level]; }

  void clear(size_t level) {
    buffer_[level].clear();
    buffer_[level].shrink_to_fit();
  }

  void free() {
    capacity_.clear();
    capacity_.shrink_to_fit();
    size_.clear();
    size_.shrink_to_fit;
    buffer_.clear();
    buffer_.shrink_to_fit();
  }

 private:
  std::vector<size_t> capacity_;
  std::vector<size_t> size_;
  std::vector<std::vector<T>> buffer_;
};
}  // namespace FMCA

#endif

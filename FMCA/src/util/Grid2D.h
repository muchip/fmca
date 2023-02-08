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
#ifndef FMCA_UTIL_GRID2D_H_
#define FMCA_UTIL_GRID2D_H_

#include <fstream>

namespace FMCA {
class Grid2D {
 public:
  Grid2D(){};
  Grid2D(const Vector &pts_min, const Vector &pts_max, Index nx, Index ny) {
    init(pts_min, pts_max, nx, ny);
  }
  void init(const Vector &pts_min, const Vector &pts_max, Index nx, Index ny) {
    pts_min_ = pts_min;
    pts_max_ = pts_max;
    nx_ = nx;
    ny_ = ny;
    P_.resize(3, nx_ * ny_);
    P_.setZero();
    hx_ = (pts_max_(0) - pts_min_(0)) / (nx_ - 1);
    hy_ = (pts_max_(1) - pts_min_(1)) / (ny_ - 1);
    Index l = 0;
    for (Index j = 0; j < ny_; ++j)
      for (Index i = 0; i < nx_; ++i, ++l)
        P_.col(l) << pts_min_(0) + hx_ * i, pts_min_(1) + hy_ * j, 0;
  }
  const Matrix &P() const { return P_; }

  void plotFunction(const std::string &fileName, const Eigen::VectorXd &f) {
    std::ofstream myfile;
    myfile.open(fileName);
    myfile << "# vtk DataFile Version 3.1\n";
    myfile << "2D grid function\n";
    myfile << "ASCII\n";
    myfile << "DATASET STRUCTURED_GRID\n";
    myfile << "DIMENSIONS " << nx_ << " " << ny_ << " " << 1 << std::endl;
    myfile << "POINTS " << P_.cols() << " FLOAT" << std::endl;
    for (auto i = 0; i < P_.cols(); ++i)
      myfile << P_(0, i) << " " << P_(1, i) << " " << f(i) << std::endl;
    myfile << "POINT_DATA " << f.size() << "\n";
    myfile << "SCALARS value FLOAT\n";
    myfile << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < f.size(); ++i) myfile << f(i) << std::endl;
    myfile.close();
  }

 private:
  Matrix P_;
  Vector pts_min_;
  Vector pts_max_;
  Scalar hx_;
  Scalar hy_;
  Index nx_;
  Index ny_;
};
}  // namespace FMCA

#endif

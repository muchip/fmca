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
#ifndef FMCA_UTIL_GRID3D_H_
#define FMCA_UTIL_GRID3D_H_

#include <fstream>

namespace FMCA {
class Grid3D {
 public:
  Grid3D(){};
  Grid3D(const Vector &pts_min, const Vector &pts_max, Index nx, Index ny,
         Index nz) {
    init(pts_min, pts_max, nx, ny, nz);
  }
  void init(const Vector &pts_min, const Vector &pts_max, Index nx, Index ny,
            Index nz) {
    pts_min_ = pts_min;
    pts_max_ = pts_max;
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    P_.resize(3, nx_ * ny_ * nz_);
    hx_ = (pts_max_(0) - pts_min_(0)) / (nx_ - 1);
    hy_ = (pts_max_(1) - pts_min_(1)) / (ny_ - 1);
    hz_ = (pts_max_(2) - pts_min_(2)) / (nz_ - 1);
    Index l = 0;
    for (Index k = 0; k < nz_; ++k)
      for (Index j = 0; j < ny_; ++j)
        for (Index i = 0; i < nx_; ++i, ++l)
          P_.col(l) << pts_min_(0) + hx_ * i, pts_min_(1) + hy_ * j,
              pts_min_(2) + hz_ * k;
  }
  const Matrix &P() const { return P_; }

  void plotFunction(const std::string &fileName, const Eigen::VectorXd &color) {
    std::ofstream myfile;
    myfile.open(fileName);
    myfile << "# vtk DataFile Version 3.1\n";
    myfile << "3D grid function\n";
    myfile << "ASCII\n";
    myfile << "DATASET STRUCTURED_GRID\n";
    myfile << "DIMENSIONS " << nx_ << " " << ny_ << " " << nz_ << std::endl;
    myfile << "POINTS " << P_.cols() << " FLOAT" << std::endl;
    for (auto i = 0; i < P_.cols(); ++i)
      myfile << P_(0, i) << " " << P_(1, i) << " " << P_(2, i) << std::endl;
    myfile << "POINT_DATA " << color.size() << "\n";
    myfile << "SCALARS value FLOAT\n";
    myfile << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < color.size(); ++i) myfile << color(i) << std::endl;
    myfile.close();
  }

 private:
  Matrix P_;
  Vector pts_min_;
  Vector pts_max_;
  Scalar hx_;
  Scalar hy_;
  Scalar hz_;
  Index nx_;
  Index ny_;
  Index nz_;
};
}  // namespace FMCA

#endif

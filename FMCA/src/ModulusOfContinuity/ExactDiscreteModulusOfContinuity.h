// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_MODULUSOFCONTINUITY_EXACTDISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_EXACTDISCRETEMODULUSOFCONTINUITY_H_

#include "../util/Macros.h"
// #include <omp.h>

namespace FMCA {

class ExactDiscreteModulusOfContinuity {
public:
  ExactDiscreteModulusOfContinuity() {}

  void init(const Matrix &P, const Matrix &f, const Scalar TX,
            const std::string dx_type = "EUCLIDEAN",
            const std::string dy_type = "EUCLIDEAN",
            const std::string trick = "NO") {
    /*P is dxn dimensional matrix (n datapoints, d dimensions)
    // f is qxn dimensional matrix (n datapoints, q dimensions)
     where f.col(i) contains function values for P.col(i)
    */

    setDistanceType(dx_, dx_type);
    setDistanceType(dy_, dy_type);

    // determine the max_distance (we will compute moc for t=
    // 0,...,max_distance) via bounding boxes

    // TX shall be optional, for now use a jolly value 0, remove trick in future
    // vers
    TX_ = TX;
    if (TX == 0) {
      TX_ = getMaxDistance(dx_, P, "TRICK");
    }
  }

  Scalar getTX() {
    if (!computed_)
      throw std::logic_error(
          "computeMocPlot not called"); // we can avoid this but for now let's
                                        // keep a single bool value
    return TX_;
  }

  Scalar getOmega(const Matrix &P, const Matrix &f, const Scalar &d) {
    // this will make use of dx, dy initialized in init (we might cast the
    // getMax function and re-use it here) computes moc exactly, for freely
    // chosen delta value t
    Scalar omega = 0;
    Index n = static_cast<Index>(P.cols());

#pragma omp parallel for reduction(max : omega)
    for (Index i = 0; i < n; i++) {
      for (Index j = 0; j < i; j++) {
        Scalar dis_x = dx_(P.col(j), P.col(i));
        if (dis_x <= d) {
          Scalar dis_y = dy_(f.col(j), f.col(i));
          if (dis_y > omega) {
            omega = dis_y;
          }
        }
      }
    }
    return omega;
  }

  void computeMocPlot(const Matrix &P, const Matrix &f, const Scalar &d) {
    // assumes the delta_steps are evenly spaced (0, d, 2d,...), rmk:
    // moc(max_distance +p)=moc(max_distance) for any p>=0 d is the delta step
    // in moc computation we discretize all pairwise distances into K_ bins of
    // width d (the zero bin, contains zero distances) e.g. d_x(i,j) belongs to
    // bin z if zd<d_x(i,j)/d<=(z+1)d

    Index n = static_cast<Index>(P.cols());
    const int num_threads = omp_get_max_threads();

    K_ = static_cast<Index>(std::ceil((TX_) / d)) + 1; //(must be an integer)
    std::vector<Vector> B_local(num_threads, Vector::Zero(K_));

// thread-local maxima: reduce across threads with element-wise max.
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();

#pragma omp for
      for (Index i = 0; i < n; i++) {

        for (Index j = 0; j < i; j++) {
          Scalar dist_x = dx_(P.col(j), P.col(i));
          Scalar dist_y = dy_(f.col(j), f.col(i));
          int bin_index = static_cast<int>(std::ceil(dist_x / d));

          // Thread-local max
          B_local[tid](bin_index) = std::max(B_local[tid](bin_index), dist_y);
        }
      }
    }

    Vector B = Vector::Zero(K_);
    tgrid_.resize(K_);
    omegat_.resize(K_);
    omegat_[0] = 0;
    tgrid_[0] = 0;

    for (int i = 1; i < K_; i++) {
      tgrid_[i] = tgrid_[i - 1] + d;
    }

    // merge thread-local vectors into final B
    for (int t = 0; t < K_; t++) {
      Scalar max_val = 0.0;
      for (int thr = 0; thr < num_threads; thr++) {
        max_val = std::max(max_val, B_local[thr](t));
      }
      B(t) = max_val;
    }

    // prefix max
    for (int i = 1; i < K_; i++) {
      omegat_[i] = std::max(omegat_[i - 1], B(i));
    }

    computed_ = true;
  }

  std::vector<Scalar> getOmegaT() {
    if (!computed_)
      throw std::logic_error("computeMocPlot not called");
    return omegat_;
  }
  std::vector<Scalar> getTGrid() {
    if (!computed_)
      throw std::logic_error("computeMocPlot not called");
    return tgrid_;
  }

private:
  std::function<Scalar(const Vector &, const Vector &)> dx_;
  std::function<Scalar(const Vector &, const Vector &)> dy_;
  Scalar TX_;
  Index K_;
  std::vector<Scalar> tgrid_;
  std::vector<Scalar> omegat_;
  bool computed_ = false;

  void
  setDistanceType(std::function<Scalar(const Vector &, const Vector &)> &df,
                  const std::string &dist_type) {

    if (dist_type == "EUCLIDEAN") {
      df = [](const Vector &x, const Vector &y) { return (x - y).norm(); };
    } else
      assert(false && "desired distance not implemented");
    return;
  }

  Scalar getMaxDistance(
      const std::function<Scalar(const Vector &, const Vector &)> &df,
      const Matrix &P, const std::string trick) {
    // assuming points are stored as columns in P
    // to be used.
    Scalar max_d = 0;

    if (trick == "TRICK") {
      // gets a quick upper bound on the max_distance, assuming L2 norm
      Scalar sum_sq = 0.0;
      // use a thread local, so that each saves the sum in its index in vector?
      // then we aggregate at the end
#pragma omp parallel for reduction(+ : sum_sq)
      for (int i = 0; i < P.rows(); i++) {
        Scalar max_r = P.row(i).maxCoeff();
        Scalar min_r = P.row(i).minCoeff();
        Scalar diff = max_r - min_r;
        sum_sq += diff * diff;
      }

      max_d = std::sqrt(sum_sq);
      return max_d;
    }

#pragma omp parallel for reduction(max : max_d)
    for (int i = 0; i < P.cols(); i++) {

      for (int j = 0; j < i; j++) {
        Scalar dist = df(P.col(j), P.col(i));
        if (dist > max_d) {
          max_d = dist;
        }
      }
    }
    return max_d;
  }
};

} // namespace FMCA
#endif

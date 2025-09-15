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
#ifndef FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITY_H_
#define FMCA_MODULUSOFCONTINUITY_DISCRETEMODULUSOFCONTINUITY_H_

namespace FMCA {

class DiscreteModulusOfContinuity {
 public:
  DiscreteModulusOfContinuity() {};

  template <typename Derived>
  void init(const Matrix &P, const Vector &f, const Scalar r, const Index R = 2,
            const Scalar TX = 1, const Index min_csize = 1,
            const std::string dist_type = "EUCLIDEAN") {
    // set all parameters
    TX_ = TX;
    r_ = r;
    R_ = R;
    K_ = std::ceil(std::log(TX / r) / std::log(R));
    min_csize_ = min_csize;
    tgrid_.resize(K_ + 1);
    omegat_.resize(K_ + 1);
    omegaNk_.resize(K_ + 1);
    XNk_indices_.resize(K_ + 1);
    setDistanceType(dist_type);
    Vector min_dist;
    {
      Derived ct(P, min_csize_);
      min_dist = minDistanceVector(ct, P);
    }
    std::cout << "fill distance: " << min_dist.maxCoeff() << std::endl;
    std::cout << "separation distance: " << min_dist.minCoeff() << std::endl;
    std::cout << "number of intervals: " << K_ + 1 << std::endl;
    {
      Derived ct(P, min_csize_);
      // set up modulus of continuity for the full set X = P using the
      // resolution r
      Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
      for (Index i = 0; i < P.cols(); ++i) {
        const std::vector<Index> nn_idcs = epsNN(ct, P, P.col(i), r_);
        for (Index j = 0; j < nn_idcs.size(); ++j)
          for (Index k = 0; k < j; ++k) {
            const Scalar xdist =
                distance_(P.col(nn_idcs[j]), P.col(nn_idcs[k]));
            assert(xdist <= 2 * r_ && "error");
            const Scalar fdist = std::abs(f(nn_idcs[j]) - f(nn_idcs[k]));
            if (xdist <= r_)
              max_quotient = max_quotient < fdist ? fdist : max_quotient;
          }
      }
      omegaNk_[0] = max_quotient;
      tgrid_[0] = r;
      omegat_[0] = omegaNk_[0];
      XNk_indices_[0].resize(P.cols());
      std::iota(XNk_indices_[0].begin(), XNk_indices_[0].end(), 0);
    }
    std::cout << "#N0=" << XNk_indices_[0].size() << std::endl;
    // compute reduced index sets using greedy method annd compute corresponding
    // moduli of continuity
    Scalar Rkr = r_;
    Matrix Pprev = P;
    for (Index k = 1; k <= K_; ++k) {
      Derived ct(Pprev, min_csize_);
      XNk_indices_[k] = greedySetCovering(ct, Pprev, Rkr);
      // fix indices to become the global indices
      for (Index j = 0; j < XNk_indices_[k].size(); ++j)
        XNk_indices_[k][j] = XNk_indices_[k - 1][XNk_indices_[k][j]];
      std::sort(XNk_indices_[k].begin(), XNk_indices_[k].end());
      Rkr *= R;
      tgrid_[k] = Rkr;
      Matrix Ploc(P.rows(), XNk_indices_[k].size());
      Vector floc(XNk_indices_[k].size());
      // set up local point set for constructing a cluster tree for epsNN
      for (Index j = 0; j < Ploc.cols(); ++j) {
        Ploc.col(j) = P.col(XNk_indices_[k][j]);
        floc(j) = f(XNk_indices_[k][j]);
      }
      Pprev = Ploc;
      Derived ctk(Ploc, min_csize_);
      Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
      for (Index i = 0; i < Ploc.cols(); ++i) {
        const std::vector<Index> nn_idcs = epsNN(ctk, Ploc, Ploc.col(i), Rkr);
        for (Index j = 0; j < nn_idcs.size(); ++j)
          for (Index k = 0; k < j; ++k) {
            const Scalar xdist =
                distance_(Ploc.col(nn_idcs[j]), Ploc.col(nn_idcs[k]));
            assert(xdist <= 2 * Rkr && "error");
            const Scalar fdist = std::abs(floc(nn_idcs[j]) - floc(nn_idcs[k]));
            if (xdist <= Rkr)
              max_quotient = max_quotient < fdist ? fdist : max_quotient;
          }
      }
      omegaNk_[k] = max_quotient;
      omegat_[k] = omegaNk_[k] > omegat_[k - 1] ? omegaNk_[k] : omegat_[k - 1];
      std::cout << "#N" << k << "=" << XNk_indices_[k].size() << std::endl;
    }
  }
  template <typename Derived>
  Scalar omega(Scalar t, const Matrix &P, const Vector &f) const {
    t = (t >= 0 ? t : 0);
    Scalar retval = 0;
    Index k = 0;
    // find interval
    if (t > TX_)
      k = K_;
    else
      while (t > tgrid_[k]) ++k;
    Matrix Ploc(P.rows(), XNk_indices_[k].size());
    Vector floc(XNk_indices_[k].size());
    // set up local point set for constructing a cluster tree for epsNN
    for (Index j = 0; j < Ploc.cols(); ++j) {
      Ploc.col(j) = P.col(XNk_indices_[k][j]);
      floc(j) = f(XNk_indices_[k][j]);
    }
    Derived ctk(Ploc, min_csize_);
    Scalar max_quotient = -1.;
#pragma omp parallel for reduction(max : max_quotient)
    for (Index i = 0; i < Ploc.cols(); ++i) {
      const std::vector<Index> nn_idcs =
          epsNN(ctk, Ploc, Ploc.col(i), tgrid_[k]);
      for (Index j = 0; j < nn_idcs.size(); ++j)
        for (Index k = 0; k < j; ++k) {
          const Scalar xdist =
              distance_(Ploc.col(nn_idcs[j]), Ploc.col(nn_idcs[k]));
          const Scalar fdist = std::abs(floc(nn_idcs[j]) - floc(nn_idcs[k]));
          if (xdist <= t)
            max_quotient = max_quotient < fdist ? fdist : max_quotient;
        }
    }
    max_quotient = max_quotient >= 0 ? max_quotient : 0;
    if (k > 0)
      max_quotient =
          max_quotient > omegat_[k - 1] ? max_quotient : omegat_[k - 1];
    return max_quotient;
  }

  const std::vector<std::vector<Index>> &XNk_indices() const {
    return XNk_indices_;
  }

  const std::vector<Scalar> &omegaNk() const { return omegaNk_; }

  const std::vector<Scalar> &omegat() const { return omegat_; }
  const std::vector<Scalar> &tgrid() const { return tgrid_; }

 private:
  void setDistanceType(const std::string &dist_type) {
    if (dist_type == "EUCLIDEAN") {
      distance_ = [](const Vector &x, const Vector &y) {
        return (x - y).norm();
      };
    } else if (dist_type == "GEODESIC") {
      distance_ = [](const Vector &x, const Vector &y) {
        return SphereClusterTree::geodesicDistance(x, y);
      };
    } else
      assert(false && "desired distance not implemented");
    return;
  }

  std::vector<std::vector<Index>> XNk_indices_;
  std::vector<Scalar> omegaNk_;
  std::vector<Scalar> tgrid_;
  std::vector<Scalar> omegat_;
  std::function<Scalar(const Vector &, const Vector &)> distance_;
  Scalar TX_;
  Scalar r_;
  Index R_;
  Index K_;
  Index min_csize_;
};
}  // namespace FMCA
#endif

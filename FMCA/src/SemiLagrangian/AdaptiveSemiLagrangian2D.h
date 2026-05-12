// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2025, Michael Multerer, Sara Avesani
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
#ifndef FMCA_SEMILAGRANGIAN_ADAPTIVESEMILAGRANGIAN2D_H_
#define FMCA_SEMILAGRANGIAN_ADAPTIVESEMILAGRANGIAN2D_H_

namespace FMCA {

/**
 *  \brief Adaptive semi-Lagrangian solver for 2D transport-type
 *         equations on scattered data, following Ling et al. and the
 *         Avesani-Giacchi-Multerer (AGM) samplet-based regularity detector.
 *
 *  Time-stepping outline:
 *    1) Compute the samplet transform of u^n.
 *    2) Run the AGM detector to obtain a leaf-wise estimate of the
 *       Jaffard exponent alpha, and flag leaves with alpha below a
 *       user-chosen threshold.
 *    3) For each data site, trace the characteristic backwards over
 *       [t^n, t^{n+1}] using a Runge-Kutta integrator and a kernel
 *       interpolation of the velocity. Foot points are projected onto
 *       the bounding box of the data, so we never extrapolate.
 *    4) Reconstruct u^n at the foot of each characteristic. On smooth
 *       sites we use the kernel reconstruction; on flagged sites we
 *       fall back to the monotone nearest-neighbour reconstruction.
 *       Optionally clamp the result to a user-supplied range, which
 *       enforces the maximum principle that holds for Burgers self-
 *       advection.
 */
class AdaptiveSemiLagrangian2D {
 public:
  enum class RKMethod { RK1, RK2, RK3, RK4 };

  using SampletTreeT = SampletTree<ClusterTree>;
  using SampletInterp = MonomialInterpolator;
  using SampletMomentsT = NystromSampletMoments<SampletInterp>;

  AdaptiveSemiLagrangian2D() = default;

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Initialise the solver.
   *
   *  \param P          data sites (2 x N)
   *  \param kernel     covariance kernel for the high-order reconstruction
   *  \param dtilde     vanishing-moment order of the samplet basis
   *  \param ridgep     Tikhonov regularisation added to the kernel diagonal
   *  \param k_low      number of neighbours used in the low-order branch
   *  \param eta        H2 admissibility parameter for the sparse compression
   *  \param threshold  a-posteriori sparsification threshold for the
   *                    samplet-compressed kernel matrix
   */
  void init(const Matrix &P, const CovarianceKernel &kernel, Index dtilde,
            Scalar ridgep = 1e-8, Index k_low = 4, Scalar eta = 0.5,
            Scalar threshold = 1e-6) {
    assert(P.rows() == 2 && "AdaptiveSemiLagrangian2D expects 2D points");
    P_ = P;
    dtilde_ = dtilde;
    bb_min_ = P_.rowwise().minCoeff();
    bb_max_ = P_.rowwise().maxCoeff();
    SampletMomentsT smom(P_, dtilde_ - 1);
    st_.init(smom, 0, P_);
    kernel_interp_.init(kernel, P_, dtilde, eta, threshold, ridgep);
    nn_interp_.init(P_, k_low);
    has_clamp_ = false;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Optional global clamp on the reconstructed values. Pass any
  /// [lo, hi] interval (e.g. the min/max of the initial condition)
  /// to enforce a discrete maximum principle.
  void setValueClamp(Scalar lo, Scalar hi) {
    clamp_lo_ = lo;
    clamp_hi_ = hi;
    has_clamp_ = true;
  }

  void disableValueClamp() { has_clamp_ = false; }

  //////////////////////////////////////////////////////////////////////////////
  /// Compression accuracy of the samplet-compressed kernel matrix
  /// (forwarded from the underlying KernelInterpolator2D /
  /// SampletKernelSolver).
  Scalar compressionError() { return kernel_interp_.compressionError(); }

  //////////////////////////////////////////////////////////////////////////////
  /// Round-trip self test for the high-order branch:
  ///   solve K c = u_test, evaluate s(P), return ||s(P) - u_test|| / ||u_test||.
  /// Should be O(threshold); if much larger, the kernel pipeline is
  /// ill-conditioned for the chosen parameters and the SL run will
  /// inject noise into u every step.
  Scalar interpolationResidual(const Vector &u_test) {
    return kernel_interp_.interpolationResidual(u_test);
  }

  //////////////////////////////////////////////////////////////////////////////
  /**
   *  \brief Run the AGM detector on @p u and return a per-data-site flag.
   *
   *  flag[i] = 1 iff the leaf cluster containing site i has an
   *  estimated Jaffard exponent below @p alpha_thr.
   *
   *  \param alpha_thr        flag a leaf if its fitted alpha is *below*
   *                          this value (small alpha = singular). Recall
   *                          that alpha_thr respresent the exponent 
   *                          (alpha + d/2) detected in the AGM. 
   *                          Setting it to 2.5 meand we flag the points
   *                          with regularity 2.5 - d/2.
   *  \param smooth_threshold relative magnitude below which the
   *                          deepest two coefficients of a leaf chain
   *                          are treated as "smooth decay" - those
   *                          leaves are then classified smooth                       
   */
  std::vector<int> detectFlags(const Vector &u, Scalar alpha_thr = 2.5,
                               Scalar smooth_threshold = 1e-6) {
    using LeafMap = std::map<const SampletTreeT *,
                             std::pair<std::vector<Scalar>, std::vector<Scalar>>>;
    const Vector tdata = st_.sampletTransform(st_.toClusterOrder(u));
    SampletCoefficientsAnalyzer<SampletTreeT> analyzer;
    analyzer.init(st_, tdata);
    LeafMap leaf_data;
    analyzer.traverseAndStackCoefficientsAndDiametersL2Norm(st_, tdata,
                                                            leaf_data);
    SlopeFitter<SampletTreeT> fitter;
    fitter.init(leaf_data, dtilde_, smooth_threshold);
    auto results = fitter.fitSlope();

    std::vector<int> flags(P_.cols(), 0);
    for (const auto &it : results) {
      const SampletTreeT *leaf = it.first;
      const Scalar alpha = it.second.get_slope();
      const int flag = (alpha < alpha_thr) ? 1 : 0;
      for (Index j = 0; j < leaf->block_size(); ++j) {
        flags[leaf->indices()[j]] = flag;
      }
    }
    return flags;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// One time step for the Burgers self-advection v = (u, 0).
  Vector burgersStep(const Vector &u, Scalar dt, RKMethod rk,
                     const std::vector<int> &flags) {
    auto v = [&](const Matrix &Q) -> Matrix {
      Matrix V(2, Q.cols());
      V.row(0) = kernel_interp_.evaluate(Q).transpose();
      V.row(1).setZero();
      return V;
    };
    return stepGeneric(u, dt, rk, v, flags);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// One time step for a user-supplied velocity field v, where v is a
  /// callable taking a 2 x M matrix of points and returning a 2 x M
  /// matrix of velocity vectors.
  template <typename VelocityFn>
  Vector stepGeneric(const Vector &u, Scalar dt, RKMethod rk,
                     const VelocityFn &v, const std::vector<int> &flags) {
    kernel_interp_.compute(u);
    Matrix ptsBack = traceCharacteristics(dt, rk, v);
    return reconstruct(u, ptsBack, flags);
  }

  //////////////////////////////////////////////////////////////////////////////
  // accessors
  const Matrix &points() const { return P_; }
  KernelInterpolator2D &kernelInterpolator() { return kernel_interp_; }
  NearestNeighborInterpolator2D &nnInterpolator() { return nn_interp_; }
  const SampletTreeT &sampletTree() const { return st_; }

 private:
  //////////////////////////////////////////////////////////////////////////////
  /// Project a point cloud onto the closed bounding box of the data.
  void projectToBB(Matrix &Q) const {
    for (Index j = 0; j < Q.cols(); ++j) {
      for (Index d = 0; d < 2; ++d) {
        if (Q(d, j) < bb_min_(d)) Q(d, j) = bb_min_(d);
        if (Q(d, j) > bb_max_(d)) Q(d, j) = bb_max_(d);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Backward Runge-Kutta tracing of the characteristics from the data
  /// sites P_ over a time step of length dt. All intermediate stages
  /// are projected back into the bounding box of the data.
  template <typename VelocityFn>
  Matrix traceCharacteristics(Scalar dt, RKMethod rk,
                              const VelocityFn &v) const {
    const Matrix &pts = P_;
    Matrix ptsBack;
    switch (rk) {
      case RKMethod::RK1: {
        Matrix v0 = v(pts);
        ptsBack = pts - dt * v0;
        break;
      }
      case RKMethod::RK2: {
        Matrix v0 = v(pts);
        Matrix p1 = pts - 0.5 * dt * v0;
        projectToBB(p1);
        Matrix v1 = v(p1);
        ptsBack = pts - dt * v1;
        break;
      }
      case RKMethod::RK3: {
        Matrix v0 = v(pts);
        Matrix p1 = pts - 0.5 * dt * v0;
        projectToBB(p1);
        Matrix v1 = v(p1);
        Matrix p2 = pts - dt * (-v0 + 2.0 * v1);
        projectToBB(p2);
        Matrix v2 = v(p2);
        ptsBack = pts - (dt / 6.0) * (v0 + 4.0 * v1 + v2);
        break;
      }
      case RKMethod::RK4: {
        Matrix v0 = v(pts);
        Matrix p1 = pts - 0.5 * dt * v0;
        projectToBB(p1);
        Matrix v1 = v(p1);
        Matrix p2 = pts - 0.5 * dt * v1;
        projectToBB(p2);
        Matrix v2 = v(p2);
        Matrix p3 = pts - dt * v2;
        projectToBB(p3);
        Matrix v3 = v(p3);
        ptsBack = pts - (dt / 6.0) * (v0 + 2.0 * v1 + 2.0 * v2 + v3);
        break;
      }
    }
    projectToBB(ptsBack);
    return ptsBack;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// Combine the high- and low-order reconstructions of @p u at the
  /// departure points @p Q according to the per-site @p flags.
  Vector reconstruct(const Vector &u, const Matrix &Q,
                     const std::vector<int> &flags) {
    Vector u_high = kernel_interp_.evaluate(Q);
    Vector u_low = nn_interp_.evaluate(u, Q);
    Vector u_new(u.size());
    for (Index i = 0; i < u.size(); ++i) {
      u_new(i) = flags[i] ? u_low(i) : u_high(i);
      if (has_clamp_) {
        if (u_new(i) < clamp_lo_) u_new(i) = clamp_lo_;
        if (u_new(i) > clamp_hi_) u_new(i) = clamp_hi_;
      }
    }
    return u_new;
  }

  //////////////////////////////////////////////////////////////////////////////
  Matrix P_;
  Vector bb_min_;
  Vector bb_max_;
  Index dtilde_ = 1;
  KernelInterpolator2D kernel_interp_;
  NearestNeighborInterpolator2D nn_interp_;
  SampletTreeT st_;
  bool has_clamp_ = false;
  Scalar clamp_lo_ = 0;
  Scalar clamp_hi_ = 0;
};

}  // namespace FMCA

#endif  // FMCA_SEMILAGRANGIAN_ADAPTIVESEMILAGRANGIAN2D_H_
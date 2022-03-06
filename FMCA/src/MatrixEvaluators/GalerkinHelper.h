// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_MATRIXEVALUATORS_GALERKINHELPER_H_
#define FMCA_MATRIXEVALUATORS_GALERKINHELPER_H_

namespace FMCA {
bool is_admissible(const TriangularPanel &el1, const TriangularPanel &el2) {
  double dist = (el1.mp_ - el2.mp_).norm() - el1.radius_ - el2.radius_;
  // dist /= (el1.radius_ > el2.radius_ ? el1.radius_ : el2.radius_);
  return (dist > 1.5);
}

////////////////////////////////////////////////////////////////////////////////
double analyticIntFS(double s, double alpha, double sx, double tx, double ux) {
  const double alphas = alpha * s;
  const double alphasx = alpha * sx;
  const double alphasmtx = alphas - tx;
  const double alphasxmtx = alphasx - tx;
  const double smsx = s - sx;
  const double onepalpha2 = 1 + alpha * alpha;
  const double p = (alpha * tx + sx) / onepalpha2;
  const double smp = s - p;
  const double q = sqrt(ux * ux + alphasxmtx * alphasxmtx / onepalpha2);
  const double sqrtonepalpha2 = sqrt(onepalpha2);
  const double firstrtxpr = sqrt(smsx * smsx + alphasmtx * alphasmtx + ux * ux);
  const double secrtxpr = sqrt(onepalpha2 * smp * smp + q * q);
  double F = 0;
  F = smsx * log(alphasmtx + firstrtxpr) - s;
  F += (alphasxmtx / sqrtonepalpha2) * log(sqrtonepalpha2 * smp + secrtxpr);
  F += 2 * ux *
       atan(((q - alphasxmtx / onepalpha2) * secrtxpr + (alphas - tx - q) * q) /
            (smp * ux));
  return F;
}
////////////////////////////////////////////////////////////////////////////////
double analyticIntFD(double s, double alpha, double stau, double sx, double tx,
                     double ux) {
  const double txmalphasx = tx - alpha * sx;
  const double onepalpha2 = 1 + alpha * alpha;
  const double p = (alpha * tx + sx) / onepalpha2;
  const double q = sqrt(ux * ux + txmalphasx * txmalphasx / onepalpha2);
  const double smp = s - p;
  const double sqrtonepalpha2 = sqrt(onepalpha2);
  const double v =
      (sqrt(onepalpha2 * smp * smp + q * q) - q) / (sqrtonepalpha2 * smp);
  // stuff related to A_1/2,  B_1/2, G_1/2
  const double ABfac2p = txmalphasx / onepalpha2 + q;
  const double ABfac2m = txmalphasx / onepalpha2 - q;
  const double ABfac1denom = (ux * ux + alpha * alpha * q * q);
  const double Afac1 = -(2 * alpha * sqrtonepalpha2 * q) / ABfac1denom;
  const double A1 = Afac1 * ABfac2p;
  const double A2 = Afac1 * ABfac2m;
  const double G1 = sqrtonepalpha2 * abs(ux) / ABfac1denom * ABfac2p;
  const double G2 = -sqrtonepalpha2 * abs(ux) / ABfac1denom * ABfac2m;
  const double sgnux = ux >= 0 ? 1 : -1;

  return sgnux * (atan((v + 0.5 * A1) / G1) - atan((v + 0.5 * A2) / G2));
}

////////////////////////////////////////////////////////////////////////////////
double analyticIntS(const TriangularPanel &el, Eigen::Vector3d x) {
  const Eigen::Vector3d sxtxux = el.cs_.transpose() * (x - el.affmap_.col(0));
  const double stau = abs(el.affmap_.col(2).dot(el.cs_.col(0)));
  const double ttau = (el.affmap_.col(2) - el.affmap_.col(1)).norm();
  const double tstar = -el.affmap_.col(1).dot(el.cs_.col(1));
  const double alpha1 = -tstar / stau;
  const double alpha2 = (ttau - tstar) / stau;
  return analyticIntFS(stau, alpha2, sxtxux(0), sxtxux(1), sxtxux(2)) -
         analyticIntFS(0, alpha2, sxtxux(0), sxtxux(1), sxtxux(2)) -
         analyticIntFS(stau, alpha1, sxtxux(0), sxtxux(1), sxtxux(2)) +
         analyticIntFS(0, alpha1, sxtxux(0), sxtxux(1), sxtxux(2));
}

////////////////////////////////////////////////////////////////////////////////
double analyticIntD(const TriangularPanel &el, Eigen::Vector3d x) {
  const Eigen::Vector3d sxtxux = el.cs_.transpose() * (x - el.affmap_.col(0));
  const double stau = abs(el.affmap_.col(2).dot(el.cs_.col(0)));
  const double ttau = (el.affmap_.col(2) - el.affmap_.col(1)).norm();
  const double tstar = -el.affmap_.col(1).dot(el.cs_.col(1));
  const double alpha1 = -tstar / stau;
  const double alpha2 = (ttau - tstar) / stau;
  if (abs(sxtxux(2)) < FMCA_ZERO_TOLERANCE)
    return double(0);
  return analyticIntFD(stau, alpha2, stau, sxtxux(0), sxtxux(1), sxtxux(2)) -
         analyticIntFD(0, alpha2, stau, sxtxux(0), sxtxux(1), sxtxux(2)) -
         analyticIntFD(stau, alpha1, stau, sxtxux(0), sxtxux(1), sxtxux(2)) +
         analyticIntFD(0, alpha1, stau, sxtxux(0), sxtxux(1), sxtxux(2));
}
} // namespace FMCA
#endif

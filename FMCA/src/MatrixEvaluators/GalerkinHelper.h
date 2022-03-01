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
  dist /= (el1.radius_ > el2.radius_ ? el1.radius_ : el2.radius_);
  return (dist > 0);
}

double analyticIntF(double s, double alpha, double sx, double tx, double ux) {
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

double analyticIntS(const TriangularPanel &el, Eigen::Vector3d x) {
  const Eigen::Vector3d sxtxux = el.cs_.transpose() * (x - el.affmap_.col(0));
  const double stau = abs(el.affmap_.col(2).dot(el.cs_.col(0)));
  const double ttau = (el.affmap_.col(2) - el.affmap_.col(1)).norm();
  const Eigen::Vector3d xstar = el.affmap_.col(0) + stau * el.cs_.col(0);
  const double tstar = -el.affmap_.col(1).dot(el.cs_.col(1));
  const double alpha1 = -tstar / stau;
  const double alpha2 = (ttau - tstar) / stau;
  return analyticIntF(stau, alpha2, sxtxux(0), sxtxux(1), sxtxux(2)) -
         analyticIntF(0, alpha2, sxtxux(0), sxtxux(1), sxtxux(2)) -
         analyticIntF(stau, alpha1, sxtxux(0), sxtxux(1), sxtxux(2)) +
         analyticIntF(0, alpha1, sxtxux(0), sxtxux(1), sxtxux(2));
}
} // namespace FMCA
#endif

// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
#include <Eigen/Dense>
#include <cstdio>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include "pardiso_interface.h"
#include "sampletMatrixGenerator.h"
////////////////////////////////////////////////////////////////////////////////
int main() {
  const unsigned int dtilde = 4;
  const double eta = 0.8;
  const unsigned int mp_deg = 6;
  const unsigned int dim = 3;
  const unsigned int n = 10000;
  const double threshold = 1e-4;
  const double ridgep = 1e-4;
  Eigen::MatrixXd P = 0.5 * (Eigen::MatrixXd::Random(dim, n).array() + 1);
  CRSmatrix S =
      sampletMatrixGenerator(P, mp_deg, dtilde, eta, threshold, ridgep);
  CRSmatrix invS = S;
  //////////////////////////////////////////////////////////////////////////////
  // PARDISO BLOCK
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "\n\nentering pardiso block\n" << std::flush;
  std::printf("ia=%p ja=%p a=%p n=%i nnz=%i\n", invS.ia.data(), invS.ja.data(),
              invS.a.data(), n, invS.ia[n]);
  std::cout << std::flush;
  pardiso_interface(invS.ia.data(), invS.ja.data(), invS.a.data(), n);
  std::cout << std::string(75, '=') << std::endl;
  Eigen::MatrixXd x(n, 10), y(n, 10), z(n, 10);
  x.setRandom();
  Eigen::VectorXd nrms = x.colwise().norm();
  for (auto i = 0; i < x.cols(); ++i)
    x.col(i) /= nrms(i);
  y.setZero();
  z.setZero();
  // compute y = S * x
  for (auto i = 0; i < n; ++i)
    for (auto j = S.ia[i]; j < S.ia[i + 1]; ++j) {
      y.row(i) += S.a[j] * x.row(S.ja[j]);
      if (i != S.ja[j])
        y.row(S.ja[j]) += S.a[j] * x.row(i);
    }
  // compute z = invS * y;
  for (auto i = 0; i < n; ++i)
    for (auto j = invS.ia[i]; j < invS.ia[i + 1]; ++j) {
      z.row(i) += invS.a[j] * y.row(invS.ja[j]);
      if (i != invS.ja[j])
        z.row(invS.ja[j]) += invS.a[j] * y.row(i);
    }
  std::cout << "inverse error:               " << (z - x).norm() / x.norm()
            << std::endl;
  return 0;
}

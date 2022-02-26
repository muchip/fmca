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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
////////////////////////////////////////////////////////////////////////////////

#include <Eigen/Dense>
#include <FMCA/H2Matrix>
//#include <FMCA/Samplets>
#include <FMCA/src/util/tictoc.hpp>

#include "../FMCA/src/H2Matrix/MomentComputer_Nystrom.h"
#include "../FMCA/src/util/Errors.h"

struct exponentialKernel {
  template <typename derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return exp(-(x - y).norm());
  }
};
// using theH2Matrix = FMCA::H2Matrix<FMCA::H2ClusterTree>;

int main(int argc, char *argv[]) {
  const auto function = exponentialKernel();
  const double eta = 0.8;
  const unsigned int mp_deg = 3;
  const unsigned int dim = atoi(argv[1]);
  tictoc T;
  std::fstream file;
  file.open("output" + std::to_string(dim) + ".txt", std::ios::out);
  file << "i          m           n     fblocks    lrblocks       nz(A)";
  file << "         mem         err\n";
  for (auto i = 2; i < 7; ++i) {
    file << i << "\t";
    const unsigned int npts = std::pow(10, i);
    const Eigen::MatrixXd P = Eigen::MatrixXd::Random(dim, npts);
    const FMCA::MomentComputer_Nystrom<const Eigen::MatrixXd> mom_comp(P);
    const FMCA::H2ClusterTree<FMCA::ClusterTree> H2CT(P, mom_comp, 1, mp_deg);
#if 0
    T.tic();
    const FMCA::NystromMatrixEvaluator<FMCA::H2ClusterTree, exponentialKernel>
        nm_eval(P, function);
    FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(H2CT, nm_eval, eta);
    const double tset = T.toc("matrix setup: ");
    {
      Eigen::VectorXd x(npts), y1(npts), y2(npts);
      double err = 0;
      double nrm = 0;
      for (auto i = 0; i < 10; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, H2CT.indices(), function, index);
        y2 = H2mat * x;
        err += (y1 - y2).squaredNorm();
        nrm += y1.squaredNorm();
      }
      err = sqrt(err / nrm);
      std::cout << "compression error: " << err << std::endl;
      // (m, n, fblocks, lrblocks, nz(A), mem)
      const std::vector<double> stats = H2mat.get_statistics();
      for (const auto &it : stats)
        file << std::setw(10) << std::setprecision(6) << it << "\t";
      file << std::setw(10) << std::setprecision(6) << err << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
#endif
  }
  file.close();

  return 0;
}

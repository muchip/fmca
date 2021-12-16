#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include "FMCA/H2Matrix"
#include "FMCA/src/H2Matrix/TensorProductInterpolation.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"

struct exponentialKernel {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<Derived> &y) const {
    return exp(-10 * (x - y).norm());
  }
};
////////////////////////////////////////////////////////////////////////////////
//#define NPTS 5000
//#define NPTS 131072
//#define NPTS 65536
//#define NPTS 32768
//#define NPTS 16384
#define NPTS 10000
//#define NPTS 4096
//#define NPTS 2048
//#define NPTS 1024
//#define NPTS 512
//#define NPTS 64
#define DIM 3
#define MPOLE_DEG 3
#define LEAFSIZE 1

int main() {
  const double eta = 0.8;
  const auto function = exponentialKernel();
  const Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  FMCA::H2ClusterTree CT(P, LEAFSIZE, MPOLE_DEG);
  unsigned int max_level = 0;
  for (const auto &n : CT)
    if (n.level() > max_level) max_level = n.level();
  Eigen::MatrixXd min_max(max_level + 1, 2);
  min_max.col(1).setZero();
  min_max.col(0).setOnes();
  min_max.col(0) *= NPTS;
  for (const auto &n : CT) {
    if (n.indices().size() > min_max(n.level(), 1))
      min_max(n.level(), 1) = n.indices().size();
    if (n.indices().size() < min_max(n.level(), 0))
      min_max(n.level(), 0) = n.indices().size();
  }
  std::cout << "max_level: " << max_level << std::endl;
  std::cout << min_max << std::endl;
  FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(P, CT, function, eta);
  H2mat.get_statistics();
  Eigen::MatrixXd K(P.cols(), P.cols());
  for (auto j = 0; j < P.cols(); ++j)
    for (auto i = 0; i < P.cols(); ++i)
      K(i, j) = function(P.col(CT.indices()[i]), P.col(CT.indices()[j]));
  std::cout << "H2-matrix compression error: "
            << (K - H2mat.full()).norm() / K.norm() << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  return 0;
}

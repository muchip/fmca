#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/H2Matrix/ChebyshevInterpolation.h"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
#define DIM 2
#define MPOLE_DEG 5
#define DTILDE 3
#define LEAFSIZE 4

using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;

int main() {
  tictoc T;
  Eigen::MatrixXd Grayscale = readMatrix("imgCompression/Lugano.txt");
  std::cout << "img data: " << Grayscale.rows() << "x" << Grayscale.cols()
            << std::endl;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Q;
  P.resize(DIM, Grayscale.rows() * Grayscale.cols());
  auto k = 0;
  for (auto i = 0; i < Grayscale.cols(); ++i)
    for (auto j = 0; j < Grayscale.rows(); ++j) {
      P(0, k) = i;
      P(1, k) = j;
      ++k;
    }
  Q.resize(3, P.cols());
  Q.setZero();
  Q.topRows(2) = P;
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE);
  T.toc("set up samplet tree: ");

  auto indices = CT.get_indices();
  Grayscale /= 255;
  Eigen::VectorXd graydata(indices.size());
  for (auto i = 0; i < indices.size(); ++i) {
    graydata(i) = Grayscale(indices[i]);
  }
  Eigen::VectorXd Tdata = ST.sampletTransform(graydata);
  ST.visualizeCoefficients(Tdata, "coeff.vtk",
                           1e-4 * Tdata.cwiseAbs().maxCoeff());
  double thres = 1e-3 * Tdata.cwiseAbs().maxCoeff();
  int zeroctr = 0;
  for (auto i = 0; i < Tdata.size(); ++i) {
    if (abs(Tdata(i)) < thres) {
      Tdata(i) = 0;
      ++zeroctr;
    }
  }
  std::cout << "original coeffs: " << Tdata.size()
            << "remaining coeffs: " << Tdata.size() - zeroctr << "compression: "
            << 1 - double(Tdata.size() - zeroctr) / Tdata.size() << std::endl;
  Tdata = ST.inverseSampletTransform(Tdata);

  std::vector<FMCA::IndexType> rev_indices;
  rev_indices.resize(indices.size());
  for (auto i = 0; i < indices.size(); ++i)
    rev_indices[indices[i]] = i;

  std::cout << "get here\n";
  Eigen::VectorXd compT = Tdata;
  for (auto i = 0; i < indices.size(); ++i) {
    compT(i) = Tdata(rev_indices[i]);
  }
  Eigen::Map<Eigen::MatrixXd> gmap(compT.data(), Grayscale.rows(), Grayscale.cols());
  Bembel::IO::print2m("Gcompress.m", "G", gmap, "w");


  FMCA::IO::plotPoints("points.vtk", CT, Q, graydata);
  return 0;
}

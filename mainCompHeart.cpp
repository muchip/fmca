#define USE_QR_CONSTRUCTION_
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/H2Matrix/ChebyshevInterpolation.h"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
#include "FMCA/src/util/print2file.hpp"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
#define DIM 3
#define MPOLE_DEG 5
#define DTILDE 3
#define LEAFSIZE 4

using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;

int main() {
  const double threshold = 0.1e-2;
  tictoc T;
  Eigen::MatrixXd Data = readMatrix("outVm.txt");
  std::cout << "img data: " << Data.rows() << "x" << Data.cols() << std::endl;
  Eigen::MatrixXd P;
  P = Data.block(0, 1, Data.rows(), 3).transpose();
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE);
  T.toc("set up samplet tree: ");

  auto indices = CT.get_indices();
  std::vector<FMCA::IndexType> rev_indices;
  rev_indices.resize(indices.size());
  for (auto i = 0; i < indices.size(); ++i) rev_indices[indices[i]] = i;
  Eigen::VectorXd data(indices.size());

  for (auto i = 0; i < indices.size(); ++i) {
    data(i) = Data(indices[i], 4);
  }
  std::cout << data.head(100) << std::endl;
  Eigen::VectorXd Tdata = ST.sampletTransform(data);
  double thres = threshold * Tdata.cwiseAbs().maxCoeff();
  std::cout << "thres: " << thres << std::endl;
  // plot relevant samplet coefficients
  ST.visualizeCoefficients(Tdata, "coeffHeart.vtk", thres);
  FMCA::IO::plotPoints("pointsHeart.vtk", CT, P, (1+1e-10) * data);
  unsigned int coeff = 0;
  for (auto i = 0; i < Tdata.size(); ++i) {
    if (abs(Tdata(i)) < thres)
      Tdata(i) = 0;
    else
      ++coeff;
  }
  std::cout << "original coeffs: " << Tdata.size() << " remaining coeffs: ("
            << coeff << ") compression: ("
            << 1 - double(coeff) / Tdata.size() << ")" << std::endl;
#if 0
  // Treddata = ST.inverseSampletTransform(Treddata);
  // Tgreendata = ST.inverseSampletTransform(Tgreendata);
  // Tbluedata = ST.inverseSampletTransform(Tbluedata);
  Eigen::VectorXd compTR = Treddata;
  Eigen::VectorXd compTG = Tgreendata;
  Eigen::VectorXd compTB = Tbluedata;
  for (auto i = 0; i < indices.size(); ++i) {
    compTR(i) = Treddata(rev_indices[i]);
    compTG(i) = Tgreendata(rev_indices[i]);
    compTB(i) = Tbluedata(rev_indices[i]);
  }
  Eigen::Map<Eigen::MatrixXd> Rmap(compTR.data(), Rchan.rows(), Rchan.cols());
  Eigen::Map<Eigen::MatrixXd> Gmap(compTG.data(), Gchan.rows(), Gchan.cols());
  Eigen::Map<Eigen::MatrixXd> Bmap(compTB.data(), Bchan.rows(), Bchan.cols());

  Bembel::IO::print2m("Rcompress.m", "R", Rmap, "w");
  Bembel::IO::print2m("Gcompress.m", "G", Gmap, "w");
  Bembel::IO::print2m("Bcompress.m", "B", Bmap, "w");
#endif
  return 0;
}

#define USE_QR_CONSTRUCTION_
#define FMCA_CLUSTERSET_
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
#include "FMCA/src/H2Matrix/TensorProductInterpolation.h"
#include "FMCA/src/util/BinomialCoefficient.h"
#include "FMCA/src/util/IO.h"
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
#define NPTS 30000
//#define NPTS 4096
//#define NPTS 2048
//#define NPTS 1024
//#define NPTS 512
//#define NPTS 64
#define DIM 3
#define MPOLE_DEG 3
#define DTILDE 3
#define LEAFSIZE 100

//#define PLOT_BOXES_
//#define TEST_H2MATRIX_
//#define TEST_COMPRESSOR_
//#define TEST_SAMPLET_TRANSFORM_
//#define TEST_VANISHING_MOMENTS_
//#define USE_BUNNY_

int main() {
  const auto function = exponentialKernel();
  const double eta = 0.6;
  const double aposteriori_threshold = 1e-10;
  const double ridge_param = 1e-3;
  //////////////////////////////////////////////////////////////////////////////
  tictoc T;
  Eigen::MatrixXd P;
  std::cout << std::string(60, '-') << std::endl;
#ifdef USE_BUNNY_
  {
    std::cout << "loading data: \n";
    Eigen::MatrixXd B = readMatrix("Points/bunnySurface.txt");
    P = B.topRows(40000).transpose();
  }
#else
  {
    std::cout << "using random points";
    P = Eigen::MatrixXd::Random(DIM, NPTS);
    P.row(2) *= 0;
    Eigen::VectorXd nrms = P.colwise().norm();
    for (auto i = 0; i < P.cols(); ++i) P.col(i) *= 1 / nrms(i);
  }
#endif
  std::cout << "data size: ";
  std::cout << P.rows() << " " << P.cols() << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "npts:        " << P.cols() << std::endl;
  std::cout << "dim:         " << P.rows() << std::endl;
  std::cout << "leaf size:   " << LEAFSIZE << std::endl;
  std::cout << "mpole deg:   " << MPOLE_DEG << std::endl;
  std::cout << "dtilde:      " << DTILDE << std::endl;
  std::cout << "eta:         " << eta << std::endl;
  std::cout << "apost thres: " << aposteriori_threshold << std::endl;
  std::cout << "ridge param: " << ridge_param << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  T.tic();
  FMCA::ClusterT CT(P, LEAFSIZE);
  T.toc("set up cluster tree: ");
  {
    std::vector<std::vector<FMCA::IndexType>> tree;
    CT.exportTreeStructure(tree);
    std::cout << "cluster structure: " << std::endl;
    std::cout << "l)\t#pts\ttotal#pts" << std::endl;
    for (auto i = 0; i < tree.size(); ++i) {
      int numInd = 0;
      for (auto j = 0; j < tree[i].size(); ++j) numInd += tree[i][j];
      std::cout << i << ")\t" << tree[i].size() << "\t" << numInd << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  T.tic();
  FMCA::H2ClusterTree H2CT(P, LEAFSIZE, MPOLE_DEG);
  T.toc("set up H2-cluster tree: ");
  {
    std::vector<std::vector<FMCA::IndexType>> tree;
    CT.exportTreeStructure(tree);
    std::cout << "cluster structure: " << std::endl;
    std::cout << "l)\t#pts\ttotal#pts" << std::endl;
    for (auto i = 0; i < tree.size(); ++i) {
      int numInd = 0;
      for (auto j = 0; j < tree[i].size(); ++j) numInd += tree[i][j];
      std::cout << i << ")\t" << tree[i].size() << "\t" << numInd << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  T.tic();
  FMCA::H2SampletTree ST(P, LEAFSIZE, DTILDE, MPOLE_DEG);
  T.toc("set up samplet tree: ");
  std::cout << std::string(60, '-') << std::endl;
  FMCA::unsymmetric_compressor_impl<FMCA::H2SampletTree> Scomp;
  FMCA::NystromMatrixEvaluator<FMCA::H2SampletTree, exponentialKernel> nm_eval;
  nm_eval.init(P, function);
  T.tic();
  Scomp.compress(ST, nm_eval, eta, aposteriori_threshold);
  T.toc("set up compressed: ");

  Eigen::MatrixXd Ktest;
  nm_eval.compute_dense_block(ST, ST, &Ktest);
#ifdef TEST_H2MATRIX_
  {
    T.tic();
    FMCA::H2Matrix<FMCA::H2ClusterTree> H2mat(P, H2CT, function, 0.2);
    T.toc("set up H2-matrix: ");
    H2mat.get_statistics();
    Eigen::MatrixXd K(P.cols(), P.cols());
    T.tic();
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) = function(P.col(H2CT.indices()[i]), P.col(H2CT.indices()[j]));
    T.toc("set up full matrix: ");
    std::cout << "H2-matrix compression error: "
              << (K - H2mat.full()).norm() / K.norm() << std::endl;
    std::cout << "H2-matrix test evaluator: " << (K - Ktest).norm() / K.norm()
              << std::endl;

    std::cout << std::string(60, '-') << std::endl;
  }
#endif
  //////////////////////////////////////////////////////////////////////////////
#ifdef TEST_COMPRESSOR_
  {
    const auto &trips = Scomp.pattern_triplets();
    std::cout << "nz per row: " << trips.size() / P.cols() << std::endl;
    std::cout << "storage sparse: "
              << double(sizeof(double) * trips.size() * 3) / double(1e9)
              << "GB\n";
    std::cout << "storage full: "
              << double(sizeof(double) * P.cols() * P.cols()) / double(1e9)
              << "GB" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    Eigen::MatrixXd K(P.cols(), P.cols());
    T.tic();
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) = function(P.col(CT.indices()[i]), P.col(CT.indices()[j]));
    ST.sampletTransformMatrix(K);
    T.toc("set up full transformed matrix: ");
    Eigen::SparseMatrix<double> S(K.rows(), K.cols());
    S.setFromTriplets(trips.begin(), trips.end());
    std::cout << "compression error: " << (S - K).norm() / K.norm()
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
#endif
  //////////////////////////////////////////////////////////////////////////////
#ifdef TEST_VANISHING_MOMENTS_
  {
    std::cout << "testing vanishing moments:\n";
    // compute the multi indices for the monomials used for vanishing moments
    FMCA::MultiIndexSet<DIM> idcs(DTILDE - 1);
    // evaluate all these monomials at the points
    Eigen::MatrixXd Pol = FMCA::momentComputer(P, CT, idcs);
    double err = 0;
    for (auto i = 0; i < Pol.rows(); ++i) {
      Pol.row(i) = ST.sampletTransform(Pol.row(i));
      err += Pol.row(i).tail(Pol.cols() - ST.get_nscalfs()).norm();
    }
    std::cout << "orthogonality error: " << err / Pol.rows() << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
#endif
  //////////////////////////////////////////////////////////////////////////////
#ifdef PLOT_BOXES_
  {
    std::vector<const ClusterT *> leafs;
    CT.getLeafIterator(leafs);
    int numInd = 0;
    for (auto i = 0; i < leafs.size(); ++i)
      numInd += (leafs[i])->get_indices().size();
    for (auto level = 0; level < 14; ++level) {
      std::vector<Eigen::Matrix3d> bbvec;
      CT.get_BboxVector(&bbvec, level);
      FMCA::IO::plotBoxes("boxes" + std::to_string(level) + ".vtk", bbvec);
    }
    std::vector<Eigen::Matrix<double, DIM, 3u>> bbvec;
    CT.get_BboxVectorLeafs(&bbvec);
    FMCA::IO::plotBoxes("boxesLeafs.vtk", bbvec);
    FMCA::IO::plotPoints("points.vtk", P);
  }
#endif
  //////////////////////////////////////////////////////////////////////////////

#ifdef TEST_SAMPLET_TRANSFORM_
  {
    eigen_assert(P.cols() <= 2e4 &&
                 "Test samplet transform only for small matrices");
    T.tic();
    const auto &trips = ST.get_transformationMatrix();
    Eigen::SparseMatrix<double> Tmat(P.cols(), P.cols());
    Tmat.setFromTriplets(trips.begin(), trips.end());
    T.toc("time samplet transform test: ");
    std::cout << "error: "
              << (Tmat * Tmat.transpose() -
                  Eigen::MatrixXd::Identity(Tmat.rows(), Tmat.cols()))
                         .norm() /
                     sqrt(Tmat.rows())
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    Eigen::MatrixXd K(P.cols(), P.cols());
    for (auto j = 0; j < P.cols(); ++j)
      for (auto i = 0; i < P.cols(); ++i)
        K(i, j) =
            function(P.col(CT.get_indices()[i]), P.col(CT.get_indices()[j]));
    Eigen::MatrixXd K2 = K;
    Eigen::MatrixXd SK = Tmat * K * Tmat.transpose();
    T.tic();
    ST.sampletTransformMatrix(K);
    T.toc("time samplet transform matrix: ");
    std::cout << "error: " << (K - SK).norm() / SK.norm() << std::endl;
    T.tic();
    ST.inverseSampletTransformMatrix(K);
    T.toc("time inverse samplet transform matrix: ");
    std::cout << "error: " << (K - K2).norm() / K2.norm() << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
#endif

  return 0;
}

#if 0
  srand(0);
  Eigen::Matrix3d rot;
  rot << 0.8047379, -0.3106172, 0.5058793, 0.5058793, 0.8047379, -0.3106172,
      -0.3106172, 0.5058793, 0.8047379;
  Eigen::MatrixXd P1 =
      2 * FMCA_PI * (Eigen::MatrixXd::Random(1, NPTS).array() + 1);
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(DIM, NPTS);
  P.row(0).array() = P1.array().cos();
  P.row(1).array() = P1.array().sin();
  P.row(2) = 0.5 * P1;
  P = rot * P;
#endif
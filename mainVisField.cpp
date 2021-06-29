#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#define USE_QR_CONSTRUCTION_
//#define FMCA_CLUSTERSET_
#define DIM 3
#define MPOLE_DEG 5
#define DTILDE 4
#define LEAFSIZE 4
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#if 0
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                         Eigen::COLAMDOrdering<int>>;
#else
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                         Eigen::MetisOrdering<int>>;
#endif
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/NormalDistribution.h"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-10 * (x - y).norm());
  }
};
////////////////////////////////////////////////////////////////////////////////
using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;
using H2ClusterT = FMCA::H2ClusterTree<ClusterT, MPOLE_DEG>;

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const double eta = 0.8;
  const double svd_threshold = 1e-6;
  const double aposteriori_threshold = 1e-10;
  const double ridge_param = 1e-2;
  const unsigned int npts = 20000;
  Eigen::MatrixXd P;
  tictoc T;
  {
    ////////////////////////////////////////////////////////////////////////////
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "loading data: ";
    Eigen::MatrixXd B = readMatrix("./Points/bunnySurface.txt");
    std::cout << "data size: ";
    std::cout << B.rows() << " " << B.cols() << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    P = B.transpose();
    P.conservativeResize(P.rows(), npts);
  }
  std::cout << std::string(60, '-') << std::endl;
  std::cout << "dim:       " << DIM << std::endl;
  std::cout << "leaf size: " << LEAFSIZE << std::endl;
  std::cout << "mpole deg: " << MPOLE_DEG << std::endl;
  std::cout << "dtilde:    " << DTILDE << std::endl;
  std::cout << "npts:      " << npts << std::endl;
  std::cout << "rparam:    " << ridge_param << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // set up cluster tree
  T.tic();
  ClusterT CT(P);
  T.toc("set up cluster tree: ");
  //////////////////////////////////////////////////////////////////////////////
  // set up H2 cluster tree
  T.tic();
  H2ClusterT H2CT(P, CT);
  T.toc("set up H2-cluster tree: ");
  {
    std::vector<std::vector<FMCA::IndexType>> tree;
    CT.exportTreeStructure(tree);
    std::cout << "cluster structure: " << std::endl;
    std::cout << "l)\t#pts\ttotal#pts" << std::endl;
    for (auto i = 0; i < tree.size(); ++i) {
      int numInd = 0;
      for (auto j = 0; j < tree[i].size(); ++j)
        numInd += tree[i][j];
      std::cout << i << ")\t" << tree[i].size() << "\t" << numInd << "\n";
    }
    std::cout << std::string(60, '-') << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  // set up samplet tree
  T.tic();
  FMCA::SampletTree<ClusterT> ST(P, CT, DTILDE, svd_threshold);
  T.toc("set up samplet tree: ");
  T.tic();
#ifdef USE_QR_CONSTRUCTION_
  std::cout << "using QR construction for samplet basis\n";
#else
  std::cout << "using SVD construction for samplet basis\n";
  std::cout << "SVD orthogonality threshold: " << svd_threshold << std::endl;
#endif
  ST.computeMultiscaleClusterBases(H2CT);
  T.toc("set up time multiscale cluster bases");
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  // set up bivariate compressor and compress matrix
  T.tic();
  FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT>> BC;
  BC.set_eta(eta);
  BC.set_threshold(aposteriori_threshold);
  std::cout << "bivariate compressor: \n";
  BC.init(P, ST, exponentialKernel());
  std::cout << std::string(60, '-') << std::endl;
  double ctime = T.toc("set up Samplet compressed matrix: ");
  std::cout << std::string(60, '-') << std::endl;
  unsigned int nz = 0;
  unsigned int nza = 0;
  unsigned int nzL = 0;
  //////////////////////////////////////////////////////////////////////////////
  // get samplet matrix and output storage and compression
  Eigen::SparseMatrix<double> W(P.cols(), P.cols());
  {
    auto trips = BC.get_Pattern_triplets();
    nz = double(trips.size()) / double(P.cols());
    nza = double(BC.get_storage_size()) / double(P.cols());
    std::cout << "nz per row:     " << nza << " / " << nz << std::endl;
    std::cout << "storage sparse: "
              << double(sizeof(double) * trips.size() * 3) / double(1e9)
              << "GB\n";
    std::cout << "storage full:   "
              << double(sizeof(double) * P.cols() * P.cols()) / double(1e9)
              << "GB" << std::endl;
    W.setFromTriplets(trips.begin(), trips.end());
  }
  std::cout << std::string(60, '-') << std::endl;
  ////////////////////////////////////////////////////////////////////////////
  // perform the Cholesky factorization of the compressed matrix
  double Chol_err = 0;
  Eigen::SparseMatrix<double> L;
  EigenCholesky solver;
  {
    T.tic();
    Eigen::SparseMatrix<double> I(P.cols(), P.cols());
    I.setIdentity();
    W += ridge_param * I;
    solver.compute(W);
    T.toc("Cholesky: ");
    std::cout << "sinfo: " << (solver.info() == Eigen::Success) << std::endl;
    std::cout << "nz Mat: " << W.nonZeros() / P.cols();
    L = solver.matrixL();
    nzL = L.nonZeros() / P.cols();
    std::cout << " nz L: " << nzL << std::endl;
    Chol_err = 0;
    {
      Eigen::VectorXd y1(P.cols());
      Eigen::VectorXd y2(P.cols());
      Eigen::VectorXd x(P.cols());
      FMCA::ProgressBar PB;
      PB.reset(10);
      for (auto i = 0; i < 10; ++i) {
        x.setRandom();
        y1 = solver.permutationPinv() *
             (L * (L.transpose() * (solver.permutationP() * x).eval()).eval())
                 .eval();
        y2 = W * x;
        Chol_err += (y1 - y2).norm() / y2.norm();
        PB.next();
      }
      Chol_err /= 10;
      std::cout << "\nCholesky decomposition error: " << Chol_err << std::endl;
    }
    W -= ridge_param * I;
  }
  FMCA::NormalDistribution ND(0, 1, 0);
  Eigen::VectorXd data;
  for (auto i = 0; i < 10; ++i) {
    data = ND.get_randMat(P.cols(), 1);
    data = ST.sampletTransform(data);
    data = solver.permutationPinv() * L * data;
    data = ST.inverseSampletTransform(data);
    FMCA::IO::plotPoints("points" + std::to_string(i) + ".vtk", CT, P, data);
  }
  return 0;
}

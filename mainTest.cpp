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
#define DIM 2
#define MPOLE_DEG 3
#define DTILDE 2
#define LEAFSIZE 4

struct exponentialKernel {
  double operator()(const Eigen::Matrix<double, DIM, 1> &x,
                    const Eigen::Matrix<double, DIM, 1> &y) const {
    return exp(-4 * (x - y).norm());
  }
};

template <typename Functor>
Eigen::VectorXd matrixMultiplier(const Eigen::MatrixXd &P,
                                 const std::vector<FMCA::IndexType> &idcs,
                                 const Functor &fun, const Eigen::VectorXd &x) {
  Eigen::VectorXd retval(x.size());
  retval.setZero();
  for (auto i = 0; i < x.size(); ++i)
    for (auto j = 0; j < x.size(); ++j)
      retval(i) += fun(P.col(idcs[i]), P.col(idcs[j])) * x(j);
  return retval;
}
using ClusterT = FMCA::ClusterTree<double, DIM, LEAFSIZE, MPOLE_DEG>;
using H2ClusterT = FMCA::H2ClusterTree<ClusterT, MPOLE_DEG>;

int main(int argc, char *argv[]) {
  const double eta = 0.8;
  const double svd_threshold = 1e-6;
  const double aposteriori_threshold = 1e-6;
  tictoc T;
  {
    std::ifstream file;
    file.open("logger.txt");
    if (!file.good()) {
      file.close();
      std::fstream newfile;
      newfile.open("logger.txt", std::fstream::out);
      newfile << std::setw(10) << "Npts" << std ::setw(5) << "dim"
              << std::setw(8) << "mpdeg" << std::setw(8) << "dtilde"
              << std::setw(6) << "eta" << std::setw(8) << "apost"
              << std::setw(8) << "svd" << std::setw(9) << "nza" << std::setw(9)
              << "nzp" << std::setw(14) << "mom ortho" << std::setw(14)
              << "error" << std::setw(12) << "ctime" << std::endl;
      newfile.close();
    }
  }
  for (int npts : {1}) {
    std::cout << "loading data: ";
    Eigen::MatrixXd B = readMatrix("./Points/bunnyProjection2D.txt");
    std::cout << "data size: ";
    std::cout << B.rows() << " " << B.cols() << std::endl;
    npts = B.rows();
    std::cout << std::string(60, '-') << std::endl;
    Eigen::MatrixXd P = B.transpose();
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "dim:       " << DIM << std::endl;
    std::cout << "leaf size: " << LEAFSIZE << std::endl;
    std::cout << "mpole deg: " << MPOLE_DEG << std::endl;
    std::cout << "dtilde:    " << DTILDE << std::endl;
    std::cout << "npts:      " << npts << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    srand(0);
    //////////////////////////////////////////////////////////////////////////////
    // set up cluster tree
    T.tic();
    ClusterT CT(P);
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
        for (auto j = 0; j < tree[i].size(); ++j) numInd += tree[i][j];
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
    {
      std::vector<const ClusterT *> leafs;
      CT.getLeafIterator(leafs);
      int numInd = 0;
      for (auto i = 0; i < leafs.size(); ++i)
        numInd += (leafs[i])->get_indices().size();
      for (auto level = 0; level < 14; ++level) {
        std::vector<Eigen::Matrix<double, DIM, 3u>> bbvec;
        CT.get_BboxVector(&bbvec, level);
        FMCA::IO::plotBoxes2D("boxes" + std::to_string(level) + ".vtk", bbvec);
      }
      std::vector<Eigen::Matrix<double, DIM, 3u>> bbvec;
      CT.get_BboxVectorLeafs(&bbvec);
      FMCA::IO::plotBoxes2D("boxesLeafs.vtk", bbvec);
      Eigen::MatrixXd Q(3, P.cols());
      Q.setZero();
      Q.topRows(2) = P;
      FMCA::IO::plotPoints("points.vtk", Q);
    }
    std::cout << std::string(60, '-') << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    // set up bivariate compressor
    T.tic();
    FMCA::BivariateCompressorH2<FMCA::SampletTree<ClusterT>> BC;
    BC.set_eta(eta);
    BC.set_threshold(aposteriori_threshold);
    std::cout << "bivariate compressor: \n";
    std::cout << "eta:       " << eta << std::endl;
    std::cout << "threshold: " << aposteriori_threshold << std::endl;
    BC.init(P, ST, exponentialKernel());
    std::cout << std::string(60, '-') << std::endl;
    double ctime = T.toc("set up Samplet compressed matrix: ");

    double nz = 0;
    double nza = 0;
    {
      auto trips = BC.get_Pattern_triplets();
      std::cout << trips.size() << " " << BC.get_storage_size() << std::endl;
      nz = double(trips.size()) / double(P.cols());
      nza = double(BC.get_storage_size()) / double(P.cols());
      std::cout << "nz per row:     " << nza << " / " << nz << std::endl;
      std::cout << "storage sparse: "
                << double(sizeof(double) * trips.size() * 3) / double(1e9)
                << "GB\n";
      std::cout << "storage full:   "
                << double(sizeof(double) * P.cols() * P.cols()) / double(1e9)
                << "GB" << std::endl;
    }
    std::cout << std::string(60, '-') << std::endl;
    double mom_err = 0;
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
      mom_err = err / Pol.rows();
      std::cout << "orthogonality error: " << mom_err << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }
    {
      T.tic();
      auto trips = BC.get_Pattern_triplets();
      Eigen::SparseMatrix<double> W(P.cols(), P.cols());
      W.setFromTriplets(
          trips.begin(), trips.end(), [](const double &a, const double &b) {
            std::cout << a << " " << b << " duplicate in triplet list\n";
            return b;
          });
      double err = 0;
      Eigen::VectorXd y1(P.cols());
      Eigen::VectorXd y2(P.cols());
      Eigen::VectorXd x(P.cols());
      FMCA::ProgressBar PB;
      PB.reset(10);
      for (auto i = 0; i < 10; ++i) {
        x.setRandom();
        y1 = matrixMultiplier(P, CT.get_indices(), exponentialKernel(), x);
        y2 = ST.inverseSampletTransform(W * ST.sampletTransform(x));
        err += (y1 - y2).norm() / y1.norm();
        PB.next();
      }
      err /= 10;
      std::cout << std::endl;
      T.toc("time error computation: ");
      std::cout << "error: " << err << std::endl;
      std::fstream newfile;
      newfile.open("logger.txt", std::fstream::app);
      newfile << std::setw(10) << npts << std ::setw(5) << DIM << std::setw(8)
              << MPOLE_DEG << std::setw(8) << DTILDE << std::setw(6) << eta
              << std::setw(8) << aposteriori_threshold << std::setw(8)
              << svd_threshold << std::setw(9) << nza << std::setw(9) << nz
              << std::setw(14) << mom_err << std::setw(14) << err
              << std::setw(12) << ctime << std::endl;
      newfile.close();
    }
  }
  return 0;
}

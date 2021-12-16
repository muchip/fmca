#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#define USE_QR_CONSTRUCTION_
#define FMCA_CLUSTERSET_
#define DIM 3
#define MPOLE_DEG 3
#define DTILDE 3
#define LEAFSIZE 100
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using EigenCholesky =
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                         Eigen::MetisOrdering<int>>;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <functional>
#include <iomanip>

#include "FMCA/H2Matrix"
#include "FMCA/Samplets"
#include "FMCA/src/util/Errors.h"
#include "FMCA/src/util/tictoc.hpp"
#include "imgCompression/matrixReader.h"
////////////////////////////////////////////////////////////////////////////////
struct exponentialKernel {
  template <typename Derived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<Derived> &y) const {
    return exp(-(x - y).norm() / sqrt(DIM));
  }
};
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  const exponentialKernel function = exponentialKernel();
  const double eta = 0.8;
  const double aposteriori_threshold = 1e-5;
  const double ridge_param = 10;
  const std::string logger = "benchmarkLogger_" + std::to_string(DIM) + ".txt";
  tictoc T;
  {
    std::ifstream file;
    file.open(logger);
    if (!file.good()) {
      file.close();
      std::fstream newfile;
      newfile.open(logger, std::fstream::out);
      newfile << std::setw(10) << "Npts" << std ::setw(5) << "dim"
              << std::setw(8) << "mpdeg" << std::setw(8) << "dtilde"
              << std::setw(6) << "eta" << std::setw(8) << "apost"
              << std::setw(9) << "nza" << std::setw(9) << "nzp" << std::setw(9)
              << "nzL" << std::setw(14) << "Comp_err" << std::setw(14)
              << "Chol_err" << std::setw(14) << "cond" << std::setw(12)
              << "ctime" << std::endl;
      newfile.close();
    }
  }
  std::cout << std::string(60, '-') << std::endl;
  //////////////////////////////////////////////////////////////////////////////
  std::cout << "loading data: \n";
  Eigen::MatrixXd B = readMatrix("Points/cross3D.txt");
  for (auto i = 4; i <= 20; ++i) {
    const unsigned int npts = 1 << i;
    Eigen::MatrixXd P = B.topRows(npts).transpose();
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "dim:       " << DIM << std::endl;
    std::cout << "leaf size: " << LEAFSIZE << std::endl;
    std::cout << "mpole deg: " << MPOLE_DEG << std::endl;
    std::cout << "dtilde:    " << DTILDE << std::endl;
    std::cout << "npts:      " << npts << std::endl;
    std::cout << std::string(60, '-') << std::endl;
#ifdef FMCA_SYMMETRIC_STORAGE_
    std::cout << "using symmetric storage!\n";
#endif
    //////////////////////////////////////////////////////////////////////////////
    // set up samplet tree
    T.tic();
    FMCA::H2SampletTree ST(P, LEAFSIZE, DTILDE, MPOLE_DEG);
    T.toc("set up samplet tree: ");
    {
      std::vector<std::vector<FMCA::IndexType>> tree;
      ST.exportTreeStructure(tree);
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
#ifdef USE_QR_CONSTRUCTION_
    std::cout << "using QR construction for samplet basis\n";
#else
    eigen_assert(false);
#endif
    T.tic();
    FMCA::unsymmetric_compressor_impl<FMCA::H2SampletTree> Scomp;
    FMCA::NystromMatrixEvaluator<FMCA::H2SampletTree, exponentialKernel>
        nm_eval;
    nm_eval.init(P, function);
    T.tic();
    Scomp.compress(ST, nm_eval, eta, aposteriori_threshold);
    double ctime = T.toc("set up Samplet compressed matrix: ");
    std::cout << std::string(60, '-') << std::endl;
    //////////////////////////////////////////////////////////////////////////////
    double nz = 0;
    double nza = 0;
    unsigned int nzL = 0;
    Eigen::SparseMatrix<double> W(P.cols(), P.cols());
    {
      const std::vector<Eigen::Triplet<double>> &trips =
          Scomp.pattern_triplets();
      Eigen::Index maxi = 0;
      Eigen::Index maxj = 0;
      for (const auto &it : trips) {
        maxi = maxi < it.row() ? it.row() : maxi;
        maxj = maxj < it.col() ? it.col() : maxj;
      }
      std::cout << "maxi: " << maxi << " maxj: " << maxj << " n: " << P.cols()
                << std::endl;
      nz = std::ceil(double(trips.size()) / double(P.cols()));
      std::cout << "nz per row:     " << nz << std::endl;
      std::cout << "storage sparse: "
                << double(sizeof(double) * trips.size() * 3) / double(1e9)
                << "GB\n";
      std::cout << "storage full:   "
                << double(sizeof(double) * P.cols() * P.cols()) / double(1e9)
                << "GB" << std::endl;
      std::cout << "beginning set from triplets ...\n" << std::flush;
      W.setFromTriplets(trips.begin(), trips.end());
      std::cout << "done.\n" << std::flush;
    }
    std::cout << std::string(60, '-') << std::endl;
    double mom_err = 0;
    ////////////////////////////////////////////////////////////////////////////
    // perform the Cholesky factorization of the compressed matrix
    double Chol_err = 0;
    double cond = 0;
#if 0
    {
      std::cout << "starting Cholesky decomposition\n";
      Eigen::SparseMatrix<double> I(P.cols(), P.cols());
      I.setIdentity();
      W += ridge_param * I;
      std::cout << "added ridge parameter!\n" << std::flush;
      EigenCholesky solver;
      T.tic();
      solver.compute(W);
      T.toc("time factorization: ");
      std::cout << "sinfo: " << solver.info() << std::endl;
      std::cout << "nz Mat: " << W.nonZeros() / P.cols();
      nzL = std::ceil(double(solver.matrixL().nestedExpression().nonZeros()) /
                      P.cols());
      std::cout << " nz L: " << nzL << std::endl;
      Chol_err = 0;
      if (npts < 1e5) {
        Eigen::VectorXd y1(P.cols());
        Eigen::VectorXd y2(P.cols());
        Eigen::VectorXd x(P.cols());
        FMCA::ProgressBar PB;
        PB.reset(10);
        for (auto i = 0; i < 10; ++i) {
          x.setRandom();
          y1 = solver.permutationPinv() *
               (solver.matrixL() * (solver.matrixL().transpose() *
                                    (solver.permutationP() * x).eval())
                                       .eval())
                   .eval();
          y2 = W.triangularView<Eigen::Lower>() * x +
               W.triangularView<Eigen::StrictlyLower>().transpose() * x;
          Chol_err += (y1 - y2).norm() / y2.norm();
          PB.next();
        }
        Chol_err /= 10;
        std::cout << "\nCholesky decomposition error: " << Chol_err
                  << std::endl;
      }
    }
#endif
    ////////////////////////////////////////////////////////////////////////////
    // perform error computation
    double err = 0;
    if (npts < 1e5) {
      T.tic();
      //Eigen::SparseMatrix<double> I(P.cols(), P.cols());
      //I.setIdentity();
      //W -= ridge_param * I;

      Eigen::VectorXd y1(P.cols());
      Eigen::VectorXd y2(P.cols());
      Eigen::VectorXd x(P.cols());
      FMCA::ProgressBar PB;
      PB.reset(10);
      for (auto i = 0; i < 20; ++i) {
        unsigned int index = rand() % P.cols();
        x.setZero();
        x(index) = 1;
        y1 = FMCA::matrixColumnGetter(P, ST.indices(), function, index);
        x = ST.sampletTransform(x);
        y2 = W.triangularView<Eigen::Lower>() * x +
             W.triangularView<Eigen::StrictlyLower>().transpose() * x;
        y2 = ST.inverseSampletTransform(y2);
        err += (y1 - y2).norm() / y1.norm();
        PB.next();
      }
      err /= 10;
      std::cout << std::endl;
      T.toc("time compression error computation: ");
      std::cout << "compression error: " << err << std::endl;
    }
    std::fstream newfile;
    newfile.open(logger, std::fstream::app);
    newfile << std::setw(10) << npts << std ::setw(5) << DIM << std::setw(8)
            << MPOLE_DEG << std::setw(8) << DTILDE << std::setw(6) << eta
            << std::setw(8) << aposteriori_threshold << std::setw(9) << nza
            << std::setw(9) << nz << std::setw(9) << nzL << std::setw(14) << err
            << std::setw(14) << Chol_err << std::setw(14) << cond
            << std::setw(12) << ctime << std::endl;
    newfile.close();
  }
  return 0;
}

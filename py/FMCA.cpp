// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2022, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the GNU Affero General Public License v3.0
// license and without any warranty, see <https://github.com/muchip/FMCA>
// for further information.
//
////////////////////////////////////////////////////////////////////////////////
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
#include <FMCA/Clustering>
#include <FMCA/CovarianceKernel>
#include <FMCA/H2Matrix>
#include <FMCA/PivotedCholesky>
#include <FMCA/Samplets>
////////////////////////////////////////////////////////////////////////////////
namespace py = pybind11;
// Samplets
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
// H2Matrix
using Interpolator = FMCA::TensorProductInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;

/**
 *  \brief wrapper class for a samplet tree (for convenience, we only use H2
 *         trees)
 *
 **/
struct pySampletTree {
  pySampletTree(){};
  pySampletTree(const FMCA::Matrix &P, FMCA::Index dtilde)
      : p_(dtilde), dtilde_(dtilde) {
    const Moments mom(P, dtilde > 0 ? dtilde : 0);
    const SampletMoments samp_mom(P, dtilde > 0 ? dtilde - 1 : 0);
    ST_.init(mom, samp_mom, 0, P);
  };
  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices().data(),
                                           ST_.indices().size());
  }
  FMCA::iVector levels() {
    std::vector<FMCA::Index> lvl = FMCA::internal::sampletLevelMapper(ST_);
    return Eigen::Map<const FMCA::iVector>(lvl.data(), lvl.size());
  }
  H2SampletTree ST_;
  FMCA::Index p_;
  FMCA::Index dtilde_;
};
////////////////////////////////////////////////////////////////////////////////

using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief wrapper class for an H2Matrix
 *
 **/
struct pyH2Matrix {
  pyH2Matrix(){};
  pyH2Matrix(const FMCA::CovarianceKernel &ker, const FMCA::Matrix &P,
             const FMCA::Index p = 3, const FMCA::Scalar eta = 0.8) {
    init(ker, P, p, eta);
  };
  void init(const FMCA::CovarianceKernel &ker, const FMCA::Matrix &P,
            const FMCA::Index p = 3, const FMCA::Scalar eta = 0.8) {
    p_ = p;
    eta_ = eta;
    const Moments mom(P, p_);
    const MatrixEvaluator mat_eval(mom, ker);
    ct_.init(mom, 0, P);
    hmat_.init(ct_, mat_eval, eta);
  }
  FMCA::Matrix statistics() const { return hmat_.get_statistics(); }

  // member variables
  H2Matrix hmat_;
  H2ClusterTree ct_;
  FMCA::Index p_;
  FMCA::Scalar eta_;
};
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief class providing Samplet kernel approximations
 *
 **/
struct pySampletKernelCompressor {
  pySampletKernelCompressor() {}
  pySampletKernelCompressor(const pySampletTree &hst,
                            const FMCA::CovarianceKernel &ker,
                            const FMCA::Matrix &P, const FMCA::Scalar eta = 0.8,
                            const FMCA::Scalar thres = 0)
      : eta_(eta), thres_(thres), n_(P.cols()) {
    init(hst, ker, P, eta, thres);
  }

  template <typename Functor>
  FMCA::Vector matrixColumnGetter(const FMCA::Matrix &P,
                                  const std::vector<FMCA::Index> &idcs,
                                  const Functor &fun, FMCA::Index colID) {
    FMCA::Vector retval(P.cols());
    retval.setZero();
    for (auto i = 0; i < retval.size(); ++i)
      retval(i) = fun(P.col(idcs[i]), P.col(idcs[colID]));
    return retval;
  }

  void init(const pySampletTree &hst, const FMCA::CovarianceKernel &ker,
            const FMCA::Matrix &P, const FMCA::Scalar eta = 0.8,
            const FMCA::Scalar thres = 0) {
    const Moments mom(P, hst.p_);
    const MatrixEvaluator mat_eval(mom, ker);
    n_ = P.cols();
    eta_ = eta;
    thres_ = thres;
    std::cout << "mpole deg:                    " << hst.p_ << std::endl;
    std::cout << "dtilde:                       " << hst.dtilde_ << std::endl;
    std::cout << "eta:                          " << eta << std::endl;
    std::cout << "thres:                        " << thres << std::endl;
    {
      FMCA::internal::SampletMatrixCompressor<H2SampletTree> scomp;
      scomp.init(hst.ST_, eta, thres);
      scomp.compress(mat_eval);
      trips_ = scomp.triplets();
    }
    std::cout << "anz:                          "
              << std::round(trips_.size() / FMCA::Scalar(P.cols()))
              << std::endl;
    FMCA::Vector x(P.cols()), y1(P.cols()), y2(P.cols());
    FMCA::Scalar err = 0;
    FMCA::Scalar nrm = 0;
    for (auto i = 0; i < 10; ++i) {
      FMCA::Index index = rand() % P.cols();
      x.setZero();
      x(index) = 1;
      y1 = matrixColumnGetter(P, hst.ST_.indices(), ker, index);
      x = hst.ST_.sampletTransform(x);
      y2.setZero();
      for (const auto &i : trips_) {
        y2(i.row()) += i.value() * x(i.col());
        if (i.row() != i.col()) y2(i.col()) += i.value() * x(i.row());
      }
      y2 = hst.ST_.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    err = sqrt(err / nrm);
    std::cout << "compression error:            " << err << std::endl;
  }

  Eigen::SparseMatrix<FMCA::Scalar> matrix() {
    Eigen::SparseMatrix<FMCA::Scalar> retval(n_, n_);
    retval.setFromTriplets(trips_.begin(), trips_.end());
    return retval;
  }

  // member variables
  std::vector<Eigen::Triplet<FMCA::Scalar>> trips_;
  FMCA::Scalar eta_;
  FMCA::Scalar thres_;
  FMCA::Index n_;
};

////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(FMCA, m) {
  m.doc() = "pybind11 FMCA plugin";  // optional module docstring
  //////////////////////////////////////////////////////////////////////////////
  // ClusterTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::ClusterTree> ClusterTree_(m, "ClusterTree");
  ClusterTree_.def(py::init<>());
  ClusterTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  m.def(
      "clusterTreeStatistics",
      [](const FMCA::ClusterTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a cluster tree");
  //////////////////////////////////////////////////////////////////////////////
  // SampletTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pySampletTree> pySampletTree_(m, "SampletTree");
  pySampletTree_.def(py::init<>());
  pySampletTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  pySampletTree_.def("indices", &pySampletTree::indices);
  pySampletTree_.def("levels", &pySampletTree::levels);

  m.def(
      "sampletTreeStatistics",
      [](const pySampletTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree.ST_, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a samplet tree");

  m.def(
      "sampletTransform",
      [](const pySampletTree &tree, const FMCA::Matrix &data) {
        return tree.ST_.sampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");

  m.def(
      "inverseSampletTransform",
      [](const pySampletTree &tree, const FMCA::Matrix &data) {
        return tree.ST_.inverseSampletTransform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Performs inverse samplet transform of data");

  m.def(
      "sampletTransformMinLevel",
      [](const pySampletTree &tree, const FMCA::Matrix &data,
         const FMCA::Index min_level) {
        FMCA::internal::SampletTransformer<H2SampletTree> s_trafo(tree.ST_,
                                                                  min_level);
        return s_trafo.transform(data);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
      "Performs samplet transform of data");
  //////////////////////////////////////////////////////////////////////////////
  // CovarianceKernel
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::CovarianceKernel> pyCovarianceKernel_(m, "CovarianceKernel");
  pyCovarianceKernel_.def(py::init<>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar>());
  pyCovarianceKernel_.def("kernelType", &FMCA::CovarianceKernel::kernelType);
  pyCovarianceKernel_.def("eval", &FMCA::CovarianceKernel::eval,
                          py::arg().noconvert(), py::arg().noconvert());
  //////////////////////////////////////////////////////////////////////////////
  // H2Matrix
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyH2Matrix> pyH2Matrix_(m, "H2Matrix");
  pyH2Matrix_.def(py::init<>());
  pyH2Matrix_.def(py::init<const FMCA::CovarianceKernel &, const FMCA::Matrix &,
                           const FMCA::Index, const FMCA::Scalar>());
  pyH2Matrix_.def("compute", &pyH2Matrix::init, py::arg().noconvert(),
                  py::arg().noconvert(), py::arg(), py::arg(),
                  "Computes the H2Matrix");
  pyH2Matrix_.def("statistics", &pyH2Matrix::statistics);
  //////////////////////////////////////////////////////////////////////////////
  // SampletCompressor
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pySampletKernelCompressor> pySampletKernelCompressor_(
      m, "SampletKernelCompressor");
  pySampletKernelCompressor_.def(py::init<>());
  pySampletKernelCompressor_.def(
      py::init<const pySampletTree &, const FMCA::CovarianceKernel &,
               const FMCA::Matrix &, const FMCA::Scalar, const FMCA::Scalar>());
  pySampletKernelCompressor_.def("compute", &pySampletKernelCompressor::init,
                                 py::arg().noconvert(), py::arg().noconvert(),
                                 py::arg().noconvert(), py::arg(), py::arg(),
                                 "Computes the compressed kernel");
  pySampletKernelCompressor_.def("matrix", &pySampletKernelCompressor::matrix,
                                 "returns the compressed kernel matrix");
  //////////////////////////////////////////////////////////////////////////////
  // pivoted Cholesky decomposition
  //////////////////////////////////////////////////////////////////////////////
   py::class_<FMCA::PivotedCholesky> pyPivotedCholesky_(m, "PivotedCholesky");
  pyPivotedCholesky_.def(py::init<>());
  pyPivotedCholesky_.def(py::init<const FMCA::CovarianceKernel &,
                                  const FMCA::Matrix &, FMCA::Scalar>());
  pyPivotedCholesky_.def("compute", &FMCA::PivotedCholesky::compute,
                         py::arg().noconvert(), py::arg().noconvert(),
                         py::arg(),
                         "Computes the pivoted Cholesky decomposition");
  pyPivotedCholesky_.def(
      "computeFullPiv", &FMCA::PivotedCholesky::computeFullPiv,
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "Computes the truncated spectral decomposition");
  pyPivotedCholesky_.def("indices", &FMCA::PivotedCholesky::indices);
  pyPivotedCholesky_.def("matrixL", &FMCA::PivotedCholesky::matrixL);
}

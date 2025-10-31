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
#include <FMCA/src/util/Tictoc.h>

#include <FMCA/Clustering>
#include <FMCA/CovarianceKernel>
#include <FMCA/H2Matrix>
#include <FMCA/LowRankApproximation>
#include <FMCA/Samplets>
#include <FMCA/Wedgelets>
////////////////////////////////////////////////////////////////////////////////
namespace py = pybind11;
// Samplets
using SampletInterpolator = FMCA::MonomialInterpolator;
using SampletMoments = FMCA::MinNystromSampletMoments<SampletInterpolator>;
using SampletTree = FMCA::SampletTree<FMCA::ClusterTree>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;
using H2SampletTreeRP = FMCA::H2SampletTree<FMCA::RandomProjectionTree>;
// H2Matrix
using Interpolator = FMCA::TotalDegreeInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using H2ClusterTree = FMCA::H2ClusterTree<FMCA::ClusterTree>;
using H2Matrix = FMCA::H2Matrix<H2ClusterTree>;
////////////////////////////////////////////////////////////////////////////////

using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using usMatrixEvaluator =
    FMCA::unsymmetricNystromEvaluator<Moments, FMCA::CovarianceKernel>;
/**
 *  \brief wrapper class for a samplet tree (for convenience, we only use H2
 *         trees)
 *
 **/
struct pySampletTree {
  pySampletTree() {};
  pySampletTree(const FMCA::Matrix &P, FMCA::Index dtilde) {
    dtilde_ = dtilde > 0 ? dtilde : 1;
    p_ = 2 * (dtilde_ - 1);
    const Moments mom(P, p_);
    const SampletMoments samp_mom(P, dtilde - 1);
    ST_.init(mom, samp_mom, 0, P);
    cluster_map_.resize(P.cols());

    for (const auto &it : ST_)
      if (it.is_root())
        for (FMCA::Index i = 0; i < it.nscalfs() + it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
      else
        for (FMCA::Index i = 0; i < it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
  };

  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices(), ST_.block_size());
  }
  FMCA::iVector levels() {
    std::vector<FMCA::Index> lvl = FMCA::internal::sampletLevelMapper(ST_);
    return Eigen::Map<const FMCA::iVector>(lvl.data(), lvl.size());
  }

  FMCA::iVector coeff2indices(FMCA::Index i) {
    const H2SampletTree &node = *(cluster_map_[i]);
    return Eigen::Map<const FMCA::iVector>(node.indices(), node.block_size());
  }

  FMCA::Matrix toClusterOrder(const FMCA::Matrix &mat) const {
    return ST_.toClusterOrder(mat);
  }

  FMCA::Matrix toNaturalOrder(const FMCA::Matrix &mat) const {
    return ST_.toNaturalOrder(mat);
  }

  FMCA::iVector adaptiveTreeLeafPartition(const FMCA::Vector &data,
                                          FMCA::Scalar thres) {
    FMCA::iVector retval(ST_.block_size());
    retval.setZero();
    std::vector<const H2SampletTree *> active_tree = adaptiveTreeSearch(
        ST_, ST_.sampletTransform(ST_.toClusterOrder(data)), thres);
    FMCA::Index cval = 1;
    for (const auto &it : active_tree)
      if (it != nullptr && !it->nSons() && it->block_size()) {
        for (FMCA::Index i = 0; i < it->block_size(); ++i)
          retval(it->indices()[i]) = cval;
        ++cval;
      }
    std::cout << "active leafs: " << cval - 1 << std::endl;
    return retval;
  }

  H2SampletTree ST_;
  FMCA::Index p_;
  FMCA::Index dtilde_;
  std::vector<const H2SampletTree *> cluster_map_;
};

/**
 *  \brief wrapper class for the wedgelet transform
 *
 **/
struct pyWedgeletTree {
  pyWedgeletTree() {};
  pyWedgeletTree(const FMCA::Matrix &P, const FMCA::Index unif_splits = 4) {
    init(P, unif_splits);
  }
  void init(const FMCA::Matrix &P, const FMCA::Index unif_splits = 4) {
    unif_splits_ = unif_splits;
    std::cout << "P: " << P.rows() << "x" << P.cols()
              << " splits: " << unif_splits << std::endl;
    WT_.init(P, unif_splits);
  }

  void computeWedges(const FMCA::Matrix &P, const FMCA::Matrix &F,
                     const FMCA::Index q = 0, const FMCA::Scalar tol = 1e-2) {
    q_ = q;
    tol_ = tol;
    WT_.computeWedges(P, F, q, tol);
  }

  FMCA::Matrix compress(const FMCA::Matrix &P, const FMCA::Matrix &F) const {
    FMCA::Matrix retval = F;
    FMCA::Index i = 0;
    for (const auto &it : WT_) {
      if (!it.nSons() && it.block_size()) {
        FMCA::MultiIndexSet<FMCA::TotalDegree> idcs(P.rows(), it.node().deg_);
        FMCA::Matrix VT(idcs.index_set().size(), it.block_size());
        for (FMCA::Index i = 0; i < it.block_size(); ++i)
          VT.col(i) =
              FMCA::internal::evalPolynomials(idcs, P.col(it.indices()[i]));
        const FMCA::Matrix eval = VT.transpose() * it.node().C_;
        for (FMCA::Index i = 0; i < it.block_size(); ++i)
          retval.row(it.indices()[i]) = eval.row(i);
      }
    }
    return retval;
  }

  FMCA::Matrix landmarks(const FMCA::Matrix &P) const {
    FMCA::Matrix retval(P.rows(), P.cols());
    FMCA::Vector hits(P.cols());
    hits.setZero();
    FMCA::Index i = 0;
    for (const auto &it : WT_) {
      if (!it.nSons() && it.block_size()) {
        retval.col(i) = P.col(it.node().landmark_);
        ++i;
        for (FMCA::Index j = 0; j < it.block_size(); ++j) {
          assert(hits(it.indices()[j]) == 0 && "duplicate index");
          hits(it.indices()[j]) = 1;
        }
        assert(hits.sum() == hits.size() && "missing index");
      }
    }
    std::cout << "non-empty leaves: " << i << std::endl;
    retval.conservativeResize(retval.rows(), i);
    return retval;
  }

  FMCA::WedgeletTree<double> WT_;
  FMCA::Scalar tol_;
  FMCA::Index q_;
  FMCA::Index unif_splits_;
};

/**
 *  \brief wrapper class for a samplet tree based on a random projection tree
 *  (for convenience, we only use H2
 *         trees)
 *
 **/
struct pySampletTreeRP {
  pySampletTreeRP() {};
  pySampletTreeRP(const FMCA::Matrix &P, FMCA::Index dtilde,
                  FMCA::Index seed = 0.) {
    dtilde_ = dtilde > 0 ? dtilde : 1;
    p_ = 2 * (dtilde_ - 1);
    const Moments mom(P, p_);
    const SampletMoments samp_mom(P, dtilde - 1);
    srand(seed);
    ST_.init(mom, samp_mom, 10, P);

    for (const auto &it : ST_)
      if (it.is_root())
        for (FMCA::Index i = 0; i < it.nscalfs() + it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
      else
        for (FMCA::Index i = 0; i < it.nsamplets(); ++i)
          cluster_map_[it.start_index() + i] = std::addressof(it);
  };

  FMCA::iVector indices() {
    return Eigen::Map<const FMCA::iVector>(ST_.indices(), ST_.block_size());
  }
  FMCA::iVector levels() {
    std::vector<FMCA::Index> lvl = FMCA::internal::sampletLevelMapper(ST_);
    return Eigen::Map<const FMCA::iVector>(lvl.data(), lvl.size());
  }

  FMCA::Matrix toClusterOrder(const FMCA::Matrix &mat) const {
    return ST_.toClusterOrder(mat);
  }

  FMCA::Matrix toNaturalOrder(const FMCA::Matrix &mat) const {
    return ST_.toNaturalOrder(mat);
  }

  FMCA::iVector coeff2indices(FMCA::Index i) {
    const H2SampletTreeRP &node = *(cluster_map_[i]);
    return Eigen::Map<const FMCA::iVector>(node.indices(), node.block_size());
  }

  FMCA::iVector level_labels(const FMCA::Index lvl) {
    FMCA::iVector retval(ST_.block_size());
    retval.setZero();
    FMCA::Index label = 1;
    FMCA::Index ctr = 0;
    std::cout << "labelling level: " << lvl << std::endl;
    for (const auto &it : ST_) {
      if (it.level() == lvl && it.block_size()) {
        for (FMCA::Index i = 0; i < it.block_size(); ++i)
          retval(it.indices()[i]) = label;
        ++ctr;
        ++label;
      }
    }
    std::cout << "nonempty clusters on level " << lvl << ": " << ctr
              << std::endl;
    return retval;
  }

  FMCA::iVector adaptiveTreeLeafPartition(const FMCA::Vector &data,
                                          FMCA::Scalar thres) {
    FMCA::iVector retval(ST_.block_size());
    retval.setZero();
    std::vector<const H2SampletTreeRP *> active_tree =
        adaptiveTreeSearch(ST_, ST_.sampletTransform(ST_.toClusterOrder(data)),
                           data.squaredNorm() * thres);
    FMCA::Index cval = 0;
    for (const auto &it : active_tree)
      if (it != nullptr && !it->nSons() && it->block_size()) {
        for (FMCA::Index i = 0; i < it->block_size(); ++i)
          retval(it->indices()[i]) = cval;
        ++cval;
      }
    std::cout << "active leafs: " << cval << std::endl;
    return retval;
  }

  H2SampletTreeRP ST_;
  FMCA::Index p_;
  FMCA::Index dtilde_;
  std::vector<const H2SampletTreeRP *> cluster_map_;
};

////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief wrapper class for an H2Matrix
 *
 **/
struct pyH2Matrix {
  pyH2Matrix(const FMCA::CovarianceKernel &ker, const FMCA::Matrix &Pr,
             const FMCA::Matrix &Pc, const FMCA::Index p = 3,
             const FMCA::Scalar eta = 0.8)
      : Pr_(Pr), Pc_(Pc), p_(p), eta_(eta) {
    ker_ = ker;
    const Moments rmom(Pr_, p_);
    const Moments cmom(Pc_, p_);
    rct_.init(rmom, 0, Pr_);
    cct_.init(cmom, 0, Pc_);
    hmat_.computePattern(rct_, cct_, eta);
  };

  FMCA::Matrix statistics() const { return hmat_.statistics(); }

  FMCA::iVector rindices() const {
    return Eigen::Map<const FMCA::iVector>(rct_.indices(), rct_.block_size());
  }
  FMCA::iVector cindices() const {
    return Eigen::Map<const FMCA::iVector>(cct_.indices(), cct_.block_size());
  }
  FMCA::Matrix action(const FMCA::Matrix &rhs) const {
    using Permutation =
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
    const Moments rmom(Pr_, p_);
    const Moments cmom(Pc_, p_);
    const usMatrixEvaluator mat_eval(rmom, cmom, ker_);
    const FMCA::Matrix srhs =
        Permutation(cindices().cast<int>()).transpose() * rhs;
    const FMCA::Matrix res = hmat_.action(mat_eval, srhs);
    return Permutation(rindices().cast<int>()) * res;
  }
  // member variables
  FMCA::CovarianceKernel ker_;
  H2Matrix hmat_;
  H2ClusterTree rct_;
  H2ClusterTree cct_;
  FMCA::Matrix Pr_;
  FMCA::Matrix Pc_;
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
                                  const FMCA::Index *idcs, const Functor &fun,
                                  FMCA::Index colID) {
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
/**
 *  \brief class providing Cholesky kernel approximations
 *
 **/
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
  m.def(
      "kNN",
      [](const FMCA::ClusterTree &tree, const FMCA::Matrix &P, FMCA::Index k) {
        return FMCA::kNN(tree, P, k);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "return the list of the k-nearest neighbours of the points in P");
  //////////////////////////////////////////////////////////////////////////////
  // SampletTree
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pySampletTree> pySampletTree_(m, "SampletTree");
  pySampletTree_.def(py::init<>());
  pySampletTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  pySampletTree_.def("indices", &pySampletTree::indices);
  pySampletTree_.def("levels", &pySampletTree::levels);
  pySampletTree_.def("adpativeTreeLeafPartition",
                     &pySampletTree::adaptiveTreeLeafPartition);
  pySampletTree_.def("coeff2indices", &pySampletTree::coeff2indices);
  //
  py::class_<pyWedgeletTree> pyWedgeletTree_(m, "WedgeletTree");
  pyWedgeletTree_.def(py::init<>());
  pyWedgeletTree_.def(py::init<const FMCA::Matrix &, FMCA::Index>());
  pyWedgeletTree_.def("init", &pyWedgeletTree::init);
  pyWedgeletTree_.def("landmarks", &pyWedgeletTree::landmarks);
  pyWedgeletTree_.def("compress", &pyWedgeletTree::compress);
  pyWedgeletTree_.def("computeWedges", &pyWedgeletTree::computeWedges);

  //
  py::class_<pySampletTreeRP> pySampletTreeRP_(m, "SampletTreeRP");
  pySampletTreeRP_.def(py::init<>());
  pySampletTreeRP_.def(
      py::init<const FMCA::Matrix &, FMCA::Index, FMCA::Index>(), py::arg("P"),
      py::arg("dtilde"), py::arg("seed") = 0);
  pySampletTreeRP_.def("indices", &pySampletTreeRP::indices);
  pySampletTreeRP_.def("levels", &pySampletTreeRP::levels);
  pySampletTreeRP_.def("level_labels", &pySampletTreeRP::level_labels);
  pySampletTreeRP_.def("adpativeTreeLeafPartition",
                       &pySampletTreeRP::adaptiveTreeLeafPartition);
  pySampletTreeRP_.def("coeff2indices", &pySampletTreeRP::coeff2indices);
  m.def(
      "sampletTreeStatistics",
      [](const pySampletTree &tree, const FMCA::Matrix &P) {
        return FMCA::clusterTreeStatistics(tree.ST_, P);
      },
      py::arg().noconvert(), py::arg().noconvert(),
      "Displays metrics of a samplet tree");
  m.def(
      "sampletTreeStatistics",
      [](const pySampletTreeRP &tree, const FMCA::Matrix &P) {
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
      "sampletTransform",
      [](const pySampletTreeRP &tree, const FMCA::Matrix &data) {
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
      "inverseSampletTransform",
      [](const pySampletTreeRP &tree, const FMCA::Matrix &data) {
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
  m.def(
      "kNN",
      [](const pySampletTree &tree, const FMCA::Matrix &P, FMCA::Index k) {
        return FMCA::kNN(tree.ST_, P, k);
      },
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "return the list of the k-nearest neighbours of the points in P");

  //////////////////////////////////////////////////////////////////////////////
  // CovarianceKernel
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::CovarianceKernel> pyCovarianceKernel_(m, "CovarianceKernel");
  pyCovarianceKernel_.def(py::init<>());
  pyCovarianceKernel_.def(py::init<const std::string &>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar>());
  pyCovarianceKernel_.def(
      py::init<const std::string &, FMCA::Scalar, FMCA::Scalar>());
  pyCovarianceKernel_.def(py::init<const std::string &, FMCA::Scalar,
                                   FMCA::Scalar, FMCA::Scalar>());
  pyCovarianceKernel_.def("kernelType", &FMCA::CovarianceKernel::kernelType);
  pyCovarianceKernel_.def("eval", &FMCA::CovarianceKernel::eval,
                          py::arg().noconvert(), py::arg().noconvert());
  //////////////////////////////////////////////////////////////////////////////
  // H2Matrix
  //////////////////////////////////////////////////////////////////////////////
  py::class_<pyH2Matrix> pyH2Matrix_(m, "H2Matrix");
  pyH2Matrix_.def(
      py::init<const FMCA::CovarianceKernel &, const FMCA::Matrix &,
               const FMCA::Matrix &, const FMCA::Index, const FMCA::Scalar>());
  pyH2Matrix_.def("statistics", &pyH2Matrix::statistics);
  pyH2Matrix_.def("action", &pyH2Matrix::action, py::arg().noconvert(),
                  "computes the matrix-vector product");
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
                                 "computes the compressed kernel");
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
      "computeOMP", &FMCA::PivotedCholesky::computeOMP, py::arg().noconvert(),
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "Computes the pivoted Cholesky decomposition using OMP");
  pyPivotedCholesky_.def("computeBiorthogonalBasis",
                         &FMCA::PivotedCholesky::computeBiorthogonalBasis,
                         "Computes the biorthogonal basis");
  pyPivotedCholesky_.def("spectralBasisWeights",
                         &FMCA::PivotedCholesky::spectralBasisWeights,
                         "returns the transformation for the spectral basis");
  pyPivotedCholesky_.def(
      "computeFullPiv", &FMCA::PivotedCholesky::computeFullPiv,
      py::arg().noconvert(), py::arg().noconvert(), py::arg(),
      "Computes the truncated spectral decomposition");
  pyPivotedCholesky_.def("indices", &FMCA::PivotedCholesky::indices);
  pyPivotedCholesky_.def("matrixL", &FMCA::PivotedCholesky::matrixL);
  pyPivotedCholesky_.def("matrixU", &FMCA::PivotedCholesky::matrixU);
  pyPivotedCholesky_.def("matrixB", &FMCA::PivotedCholesky::matrixB);
  pyPivotedCholesky_.def("eigenvalues", &FMCA::PivotedCholesky::eigenvalues);
  //////////////////////////////////////////////////////////////////////////////
  // FALKON
  //////////////////////////////////////////////////////////////////////////////
  py::class_<FMCA::FALKON> pyFALKON_(m, "FALKON");
  pyFALKON_.def(py::init<>());
  pyFALKON_.def(py::init<const FMCA::CovarianceKernel &, const FMCA::Matrix &,
                         FMCA::Index, FMCA::Scalar>());
  pyFALKON_.def("init", &FMCA::FALKON::init, py::arg().noconvert(),
                py::arg().noconvert(), py::arg(), py::arg(),
                "Initializes FALKON by computing centers and matrices");
  pyFALKON_.def(
      "computeAlpha", &FMCA::FALKON::computeAlpha, py::arg().noconvert(),
      py::arg(),
      "computes coefficients for the initialized centers and the given RHS");
  pyFALKON_.def("indices", &FMCA::FALKON::indices);
  pyFALKON_.def("matrixC", &FMCA::FALKON::matrixC);
  pyFALKON_.def("matrixKPC", &FMCA::FALKON::matrixKPC);
}

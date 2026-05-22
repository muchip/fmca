#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../FMCA/CovarianceKernel"
#include "../FMCA/KernelInterpolation"
#include "../FMCA/Samplets"
#include "../FMCA/src/Samplets/samplet_matrix_compressor.h"
#include "../FMCA/src/util/Tictoc.h"

using Cholesky = Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<FMCA::Scalar>,
                                             Eigen::Upper>;
using Interpolator = FMCA::TotalDegreeInterpolator;
using SampletInterpolator = FMCA::MonomialInterpolator;
using Moments = FMCA::NystromMoments<Interpolator>;
using SampletMoments = FMCA::NystromSampletMoments<SampletInterpolator>;
using MatrixEvaluator = FMCA::NystromEvaluator<Moments, FMCA::CovarianceKernel>;
using H2SampletTree = FMCA::H2SampletTree<FMCA::ClusterTree>;

template <typename T>
void PivotedCholesky(const T& K, FMCA::Matrix* L,
                     std::vector<FMCA::Index>* idcs, FMCA::Scalar tol = 1e-3,
                     FMCA::Index max_cols = 1000) {
  FMCA::Vector D = K.diagonal();
  FMCA::Index pivot = 0;
  FMCA::Scalar tr = 0;
  L->resize(K.rows(), max_cols);
  idcs->resize(max_cols);
  tr = D.sum();
  // we guarantee the error tr(K-LL^T)/tr(K) < tol
  tol *= tr;
  // perform pivoted Cholesky decomposition
  std::cout << "N: " << K.rows() << " max number of cols: " << max_cols
            << std::endl
            << "rel tol: " << tol << " initial trace: " << tr << std::endl;
  FMCA::Index step = 0;
  while ((step < max_cols) && (tol < tr)) {
    D.maxCoeff(&pivot);
    (*idcs)[step] = pivot;
    // get new column from K
    L->col(step) = K.col(pivot);
    // update column with the current matrix Lmatrix_
    L->col(step) -= L->leftCols(step) * L->row(pivot).head(step).transpose();
    if ((*L)(pivot, step) <= 0) {
      std::cout << "breaking with non positive pivot\n";
      break;
    }
    L->col(step) /= sqrt((*L)(pivot, step));
    // update the diagonal and the trace
    D.array() -= L->col(step).array().square();
    // compute the trace of the Schur complement
    tr = D.sum();
    ++step;
  }
  std::cout << "steps: " << step << " trace error: " << tr << std::endl;
  // crop L, indices to their actual size
  L->conservativeResize(L->rows(), step);
  idcs->resize(step);
  return;
}

#define NPTS 200000
#define DIM 2
#define USE_PIVOTED_CHOLESKY

FMCA::Scalar u_true(FMCA::Scalar x, FMCA::Scalar y) {
  // Affine trend
  FMCA::Scalar val = 0.30 - 0.20 * x + 0.10 * y;
  // bumps
  val += 0.90 *
         std::exp(-40.0 * ((x - 0.25) * (x - 0.25) + (y - 0.30) * (y - 0.30)));
  val += -0.70 *
         std::exp(-60.0 * ((x - 0.72) * (x - 0.72) + (y - 0.65) * (y - 0.65)));
  val += 0.45 *
         std::exp(-50.0 * ((x - 0.55) * (x - 0.55) + (y - 0.20) * (y - 0.20)));
  // smooth oscillation
  val += 0.08 * std::sin(4.0 * FMCA_PI * x) * std::cos(3.0 * FMCA_PI * y);
  return val;
}

int main() {
  FMCA::Tictoc T;
  std::cout << "N = " << NPTS << ", d = " << DIM << std::endl;
  const FMCA::CovarianceKernel function("multiquadric", 1.0, 0.5);
  const FMCA::Matrix P = 0.5 * (FMCA::Matrix::Random(DIM, NPTS).array() + 1);
  // Training data
  FMCA::Vector f(NPTS);
  for (FMCA::Index i = 0; i < NPTS; ++i) {
    f(i) = u_true(P(0, i), P(1, i));
  }

  const FMCA::Scalar threshold = 0;
  const FMCA::Scalar eta = 0.5;
  const FMCA::Index dtilde = 8;
  const FMCA::Index mpole_deg = 2 * (dtilde - 1);

  std::cout << "dtilde:                       " << dtilde << std::endl;
  std::cout << "mpole_deg:                    " << mpole_deg << std::endl;
  std::cout << "eta:                          " << eta << std::endl;
  std::cout << "threshold:                    " << threshold << std::endl;

  // Build samplet tree
  T.tic();
  const Moments mom(P, mpole_deg);
  const Moments Pmom(P, 0);
  const MatrixEvaluator mat_eval(mom, function);
  const SampletMoments samp_mom(P, dtilde - 1);
  H2SampletTree hst(mom, samp_mom, 0, P);
  T.toc("samplet tree:                ");

  // Polynomial block UP = [S; 0]
  const FMCA::Matrix Pol = Pmom.moment_matrix(hst);                 // mq x N
  const FMCA::Matrix TPol = hst.sampletTransform(Pol.transpose());  // N x mq
  const FMCA::Index mq = Pmom.interp().idcs().index_set().size();
  std::cout << "mq:                           " << mq << std::endl;
  std::cout << "Pol size:                     " << Pol.rows() << " x "
            << Pol.cols() << std::endl;
  const FMCA::Matrix S_block = TPol.topRows(mq);  // mq x mq
  const FMCA::Matrix STS = S_block.transpose() * S_block;
  std::cout << "UP=[S;0] error:               "
            << (STS - TPol.transpose() * TPol).norm() / TPol.norm()
            << std::endl;

  // Compress kernel matrix in samplet coordinates
  T.tic();
  FMCA::internal::SampletMatrixCompressor<H2SampletTree> Scomp;
  Scomp.init(hst, eta, 100 * FMCA_ZERO_TOLERANCE);
  T.toc("planner:                     ");
  T.tic();
  Scomp.compress(mat_eval);
  T.toc("compressor:                  ");
  T.tic();
  const auto& ap_trips = Scomp.triplets();
  std::cout << "anz (a-priori):               "
            << std::round(ap_trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  T.toc("triplets:                    ");
  T.tic();
  const auto& trips = Scomp.aposteriori_triplets_fast(threshold);
  std::cout << "anz (a-posteriori):           "
            << std::round(trips.size() / FMCA::Scalar(NPTS)) << std::endl;
  T.toc("a-posteriori:                ");
  Eigen::SparseMatrix<FMCA::Scalar> Smat(NPTS, NPTS);
  Smat.setFromTriplets(trips.begin(), trips.end());

  // Compression error check
  {
    FMCA::Scalar err = 0.0, nrm = 0.0;
    for (int i = 0; i < 10; ++i) {
      FMCA::Index idx = rand() % NPTS;
      FMCA::Vector x = FMCA::Vector::Zero(NPTS);
      x(idx) = 1.0;
      FMCA::Vector col = function.eval(P, P.col(hst.indices()[idx]));
      FMCA::Vector y1 =
          col(Eigen::Map<const FMCA::iVector>(hst.indices(), NPTS));
      x = hst.sampletTransform(x);
      FMCA::Vector y2 = FMCA::Vector::Zero(NPTS);
      for (const auto& t : trips) {
        y2(t.row()) += t.value() * x(t.col());
        if (t.row() != t.col()) y2(t.col()) += t.value() * x(t.row());
      }
      y2 = hst.inverseSampletTransform(y2);
      err += (y1 - y2).squaredNorm();
      nrm += y1.squaredNorm();
    }
    std::cout << "compression error:            " << std::sqrt(err / nrm)
              << std::endl;
  }

  Eigen::SparseMatrix<FMCA::Scalar> Kpsi =
      Smat.block(mq, mq, NPTS - mq, NPTS - mq);
  FMCA::Vector Uf = hst.sampletTransform(hst.toClusterOrder(f));
  FMCA::Vector f_P = Uf.head(mq);
  FMCA::Vector f_Psi = Uf.tail(NPTS - mq);
  FMCA::Vector c_Psi;

  //////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_PIVOTED_CHOLESKY

  T.tic();
  Eigen::SparseMatrix<FMCA::Scalar> Ksym = Kpsi.selfadjointView<Eigen::Upper>();
  T.toc("build symmetric K_psi:       ");
  const FMCA::Scalar pchol_tol = 1e-14;  // trace tolerance, smaller => bigger rank
  const FMCA::Index pchol_max_cols = 20000;

  FMCA::Matrix L;
  std::vector<FMCA::Index> pivots;
  T.tic();
  PivotedCholesky(Ksym, &L, &pivots, pchol_tol, pchol_max_cols);
  T.toc("pivoted Cholesky:            ");
  std::cout << "rank (cols of L):             " << L.cols() << std::endl;

  // =========================================================================
  //  (A) QR + two triangular solves
  //  (B) Normal equations on L^T L, squares cond
  //  (C) Woodbury / SMW with small ridge  => Tikhonov-regularized solve
  //                                          c = (1/eps)(f - L (eps I + L^T
  //                                          L)^{-1} L^T f)
  // =========================================================================
  const FMCA::Index r = L.cols();
  std::cout << "\n--- comparing three solves on the same L (rank " << r
            << ") ---" << std::endl;

  //----------- (A) QR-based Moore-Penrose --------------------------------
  T.tic();
  Eigen::HouseholderQR<FMCA::Matrix> qr(L);
  const FMCA::Matrix R =
      qr.matrixQR().topLeftCorner(r, r).triangularView<Eigen::Upper>();
  FMCA::Vector p = qr.householderQ().transpose() * f_Psi;      // length m
  p.conservativeResize(r);                                     // top r
  FMCA::Vector u = R.triangularView<Eigen::Upper>().solve(p);  // R u = p
  FMCA::Vector v =
      R.transpose().triangularView<Eigen::Lower>().solve(u);  // R^T v = u
  FMCA::Vector vhat = FMCA::Vector::Zero(L.rows());
  vhat.head(r) = v;
  FMCA::Vector c_Psi_qr = qr.householderQ() * vhat;
  T.toc("(A) QR solve:                ");
  {
    FMCA::Vector res = Kpsi.selfadjointView<Eigen::Upper>() * c_Psi_qr - f_Psi;
    std::cout << "    residual:                 " << res.norm() / f_Psi.norm()
              << std::endl;
  }

  //----------- (B) Normal equations: c = L (L^T L)^{-2} L^T f -----------
  T.tic();
  FMCA::Matrix LtL = L.transpose() * L;  // r x r, cond ~ cond(L)^2
  Eigen::LLT<FMCA::Matrix> llt_LtL(LtL);
  FMCA::Vector z = L.transpose() * f_Psi;  // r
  FMCA::Vector y1 = llt_LtL.solve(z);      // (L^T L)^{-1} z
  FMCA::Vector y2 = llt_LtL.solve(y1);     // (L^T L)^{-2} z
  FMCA::Vector c_Psi_ne = L * y2;
  T.toc("(B) L^T L normal eqns:       ");
  {
    FMCA::Vector res = Kpsi.selfadjointView<Eigen::Upper>() * c_Psi_ne - f_Psi;
    std::cout << "    residual:                 " << res.norm() / f_Psi.norm()
              << std::endl;
  }

  //----------- (C) Woodbury / Sherman-Morrison on (eps I + L L^T) -------
  const FMCA::Scalar eps = 1e-7;
  T.tic();
  FMCA::Matrix M = LtL;
  M.diagonal().array() += eps;  // eps I_r + L^T L
  Eigen::LLT<FMCA::Matrix> llt_M(M);
  FMCA::Vector zw = L.transpose() * f_Psi;  // r
  FMCA::Vector w = llt_M.solve(zw);         // (eps I + L^T L)^{-1} z
  FMCA::Vector c_Psi_wb = (f_Psi - L * w) / eps;
  T.toc("(C) Woodbury:                ");
  {
    FMCA::Vector LLt_c = L * (L.transpose() * c_Psi_wb);
    FMCA::Vector res_C = eps * c_Psi_wb + LLt_c - f_Psi;
    std::cout << "    Woodbury residual on (lambda I + L L^T): "
              << res_C.norm() / f_Psi.norm() << std::endl;
  }

  // Choose which one feeds the downstream interpolation.
  c_Psi = c_Psi_wb;

#else
  {
    Eigen::SparseMatrix<FMCA::Scalar> I(NPTS - mq, NPTS - mq);
    I.setIdentity();
    Kpsi += I * 1e-8;
  }
  T.tic();
  Cholesky llt;
  llt.compute(Kpsi);
  std::cout << "Cholesky:                     "
            << (llt.info() == Eigen::Success ? "SUCCESS" : "FAILED")
            << std::endl;
  T.toc("Cholesky:                    ");
  if (llt.info() != Eigen::Success) return 1;

  T.tic();
  c_Psi = llt.solve(f_Psi);
  T.toc("solve:                       ");

  FMCA::Vector r = Kpsi.selfadjointView<Eigen::Upper>() * c_Psi - f_Psi;
  std::cout << "Cholesky residual:            " << r.norm() / f_Psi.norm()
            << std::endl;
#endif

  // c_samplet = [0; c_Psi]
  FMCA::Vector c_samplet(NPTS);
  c_samplet.head(mq).setZero();
  c_samplet.tail(NPTS - mq) = c_Psi;
  FMCA::Vector c = hst.toNaturalOrder(hst.inverseSampletTransform(c_samplet));
  // S^T S d = S^T (f_P - K_{P,Psi} c_Psi)
  Eigen::SparseMatrix<FMCA::Scalar> K_PPsi = Smat.block(0, mq, mq, NPTS - mq);
  FMCA::Vector KPPsi_cPsi = K_PPsi * c_Psi;
  FMCA::Vector rhs_d = S_block.transpose() * (f_P - KPPsi_cPsi);
  FMCA::Vector d = STS.inverse() * rhs_d;
  std::cout << "d recovered:                  " << d.transpose() << std::endl;

  //////////////////////////////////////////////////////////// Evaluation
  T.tic();
  const FMCA::Index NEVAL = 500000;
  const FMCA::Matrix P_eval =
      0.5 * (FMCA::Matrix::Random(DIM, NEVAL).array() + 1);

  FMCA::MultipoleFunctionEvaluator mfe;
  mfe.init(function, P, P_eval, eta, mpole_deg);
  FMCA::Vector mu_kernel = mfe.evaluate(P, P_eval, c);

  const Moments mom_eval(P_eval, mpole_deg);
  const Moments Pmom_eval(P_eval, 0);
  const MatrixEvaluator mat_eval_eval(mom_eval, function);
  const SampletMoments samp_mom_eval(P_eval, dtilde - 1);
  H2SampletTree hst_eval(mom_eval, samp_mom_eval, 0, P_eval);
  const FMCA::Matrix Pol_eval = Pmom_eval.moment_matrix(hst_eval);
  // Pol_eval.transpose() * d is in hst_eval cluster order -> permute to natural
  FMCA::Vector mu_poly_clust = Pol_eval.transpose() * d;
  FMCA::Vector mu_poly(NEVAL);
  for (FMCA::Index i = 0; i < NEVAL; ++i)
    mu_poly(hst_eval.indices()[i]) = mu_poly_clust(i);
  FMCA::Vector mu = mu_kernel + mu_poly;

  FMCA::Vector f_true(NEVAL);
  for (FMCA::Index i = 0; i < NEVAL; ++i) {
    f_true(i) = u_true(P_eval(0, i), P_eval(1, i));
  }

  std::cout << "Relative L2 prediction error: "
            << (mu - f_true).norm() / f_true.norm() << std::endl;
  T.toc("prediction:                  ");

  return 0;
}
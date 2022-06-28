// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2021, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
#ifndef FMCA_INTERPOLATORS_WEIGHTEDTOTALDEGREEINTERPOLATOR2_H_
#define FMCA_INTERPOLATORS_WEIGHTEDTOTALDEGREEINTERPOLATOR2_H_
#include "../util/HaltonSet.h"
#include"../util/MultiIndexSet.h"
#include "LejaPoints.h"
#include <Eigen/LU>
namespace FMCA {

class WeightedTotalDegreeInterpolator {
public:
  typedef FloatType ValueType;
  typedef Eigen::Matrix<FloatType, Eigen::Dynamic, 1> eigenVector;
  typedef Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;

  void init(IndexType deg, const std::vector<FloatType> &weights) {
    dim_ = weights.size();
    deg_ = deg;
    idcs_.init(dim_, deg_, weights);
    TD_xi_.resize(dim_, idcs_.index_set().size());
    V_.resize(idcs_.index_set().size(), idcs_.index_set().size());

    const unsigned int num_pts = 810*dim_;
    const unsigned int S = 10;

    FMCA::HaltonSet<S> halton_pts(dim_);
    Eigen::MatrixXd P (num_pts,dim_);
    for (auto i = 0; i < P.rows(); ++i) {
      P.row(i) = halton_pts.EigenHaltonVector();
      halton_pts.next();
    };

    ////Determine tensor product interpolation points
    Eigen::MatrixXd Tr;
    Tr = P.transpose();
    Eigen::MatrixXd V_init (num_pts, idcs_.index_set().size());
    Eigen::MatrixXd VD (num_pts,deg_+1);

    for (auto i = 0; i < num_pts; ++i){
      V_init.row(i) = evalPolynomials(Tr.col(i)).transpose();
    };

    //Hausholder QR decomposition
    Eigen::MatrixXd R;
    Eigen::MatrixXd R_new (idcs_.index_set().size(),idcs_.index_set().size());
    Eigen::MatrixXd R_inv;
    auto qr = V_init.householderQr();
    R = qr.matrixQR().triangularView<Eigen::Upper>();

    for (int i = 0; i<idcs_.index_set().size(); ++i){
      for (int j = 0; j<idcs_.index_set().size(); ++j){
        R_new(i,j) = R(i,j);
      };
    };
    R_inv = R_new.inverse();

    Eigen::MatrixXd V1;
    V1 = (V_init*R_inv).transpose();

    Eigen::MatrixXd IND (idcs_.index_set().size(),1);

    Eigen::MatrixXd prj;
    int indi = 0;
    double maxi;
    Eigen::MatrixXd normi (num_pts,1);
    double den; 
    double num;
    for(int e = 0; e < idcs_.index_set().size(); ++e){
      for (int i = 0; i < num_pts; ++i){
        normi(i) = V1.col(i).norm();
      };

      maxi = .0;
      for (int i = 0; i < num_pts; ++i){
        if (normi(i) > maxi){
          maxi = normi(i);
          indi = i;
        };
      };
        
      auto const J = V1.col(indi);
      auto A = V1;
      for (int i = 0; i < num_pts; ++i){
        den = J.transpose()*J;
        num = J.transpose()*A.col(i);
        prj = J*(num/den);
        A.col(i) =  A.col(i) - prj;
      };
      V1 = A;
      IND(e) = indi;
    };


    Eigen::MatrixXd p2 (idcs_.index_set().size(), dim_);
      for (int i = 0; i < idcs_.index_set().size(); ++i){
        for (int j = 0; j < dim_; ++j)
          p2(i,j) = P(IND(i),j);
      };
    TD_xi_ = p2.transpose();
      
    Eigen::MatrixXd V_ (idcs_.index_set().size(), idcs_.index_set().size());
    for (int i = 0; i < idcs_.index_set().size(); ++i){
      V_.row(i) = V_init.row(IND(i));
    };

    invV_ = V_.inverse();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  ////////////Legendre Polynomial/////////////////////////////////////////////// 
  template <typename Derived>
  eigenMatrix
  evalLegendrePolynomials1D(const Eigen::MatrixBase<Derived> &pt) const {
    eigenMatrix retval(pt.rows(), deg_ + 1);
    eigenVector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setZero();
    P1.setOnes();
    retval.col(0) = P1;
    for (auto i = 1; i <= deg_; ++i) {
      retval.col(i) = ValueType(2 * i - 1) / ValueType(i) *
                          (2 * pt.array() - 1) * P1.array() -
                      ValueType(i - 1) / ValueType(i) * P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  ////////////Chebychev Polynomial/////////////////////////////////////////////// 
  template <typename Derived>
  eigenMatrix
  evalChebychevPolynomials1D(const Eigen::MatrixBase<Derived> &pt) const {
    eigenMatrix retval(pt.rows(), deg_ + 1);
    eigenVector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setOnes();
    P1 = 2*pt.array() - 1;
    retval.col(0) = P0;
    retval.col(1) = P1;
    for (auto i = 2; i <= deg_; ++i) {
      retval.col(i) = (4 * pt.array() - 2) * P1.array() - P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  //// Normal tensor product
  template <typename Derived>
    eigenVector evalPolynomials(const Eigen::MatrixBase<Derived> &pt) const {
    eigenVector retval(idcs_.index_set().size());
  ////////////////////////////////////////////////////////////////////////////
  //// Choose the polynomial
    // eigenMatrix p_values = evalLegendrePolynomials1D(pt);
    eigenMatrix p_values = evalChebychevPolynomials1D(pt);
  ////////////////////////////////////////////////////////////////////////////
    retval.setOnes();
    IndexType k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < dim_; ++i)
        retval(k) *= p_values(i, it[i]);
      ++k;
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  const eigenMatrix &Xi() const { return TD_xi_; }
  const eigenMatrix &invV() const { return invV_; }
  const eigenMatrix &V() const { return V_; }

private:
  MultiIndexSet<WeightedTotalDegree> idcs_;
  eigenMatrix TD_xi_;
  eigenMatrix invV_;
  eigenMatrix V_;
  IndexType dim_;
  IndexType deg_;
};
} // namespace FMCA
#endif

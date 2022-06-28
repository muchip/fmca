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
////////////////////////////////////////////////////////////////////////////////
// #include <cmath>
#include </Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1/math.h>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACK

// #define LAPACK_COMPLEX_CUSTOM
// #define lapack_complex_float std::complex<float>
// #define lapack_complex_double std::complex<double>

#include <iostream>
#include <Eigen/Dense>
#include "../util/HaltonSet.h"
#include"../util/MultiIndexSet.h"
#include "LejaPoints.h"
#include <Eigen/LU>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
using namespace std;

// typedef FloatType ValueType;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> eigenVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;


  // template <typename Derived>
  // eigenMatrix evalChebychevPolynomials1D(const Eigen::MatrixBase<Derived> &pt) {
  //   const unsigned int deg_ = 6;
  //   eigenMatrix retval(pt.rows(), deg_ + 1);
  //   eigenVector P0, P1;
  //   P0.resize(pt.rows());
  //   P1.resize(pt.rows());
  //   P0.setOnes();
  //   P1 = 2*pt.array() - 1;
  //   retval.col(0) = P0;
  //   retval.col(1) = P1;
  //   for (auto i = 2; i < deg_ + 1; ++i) {
  //     retval.col(i) = (4.0 * pt.array() - 2.0) * P1.array() - P0.array();
  //     P0 = P1;
  //     P1 = retval.col(i);
  //     // L2-normalize
  //     retval.col(i) *= sqrt(2 * i + 1);
  //   }
  //   return retval;
  // }

 template <typename Derived>
  eigenMatrix evalLegendrePolynomials1D(const Eigen::MatrixBase<Derived> &pt) {
    const unsigned int deg_ = 13;
    eigenMatrix retval(pt.rows(), deg_ + 1);
    eigenVector P0, P1;
    P0.resize(pt.rows());
    P1.resize(pt.rows());
    P0.setZero();
    P1.setOnes();
    retval.col(0) = P1;
    for (auto i = 1; i <= deg_; ++i) {
      retval.col(i) = double(2 * i - 1) / double(i) *
                          (2 * pt.array() - 1) * P1.array() -
                      double(i - 1) / double(i) * P0.array();
      P0 = P1;
      P1 = retval.col(i);
      // L2-normalize
      retval.col(i) *= sqrt(2 * i + 1);
    }
    return retval;
  }


template <typename Derived>
eigenVector evalPolynomials(const Eigen::MatrixBase<Derived> &pt) {
    const unsigned int dim = 2;
    const unsigned int deg_ = 13;
    std::vector<double> weights(dim, 1);
    FMCA::MultiIndexSet<FMCA::WeightedTotalDegree> idcs_;
    idcs_.init(dim, deg_, weights);
    eigenVector retval(idcs_.index_set().size());
    eigenMatrix p_values = evalLegendrePolynomials1D(pt);
    retval.setOnes();
    int k = 0;
    for (const auto &it : idcs_.index_set()) {
      for (auto i = 0; i < dim; ++i)
        retval(k) *= p_values(i, it[i]);
      ++k;
    }
    return retval;
  }


int main(int argc, char *argv[]) {

      const unsigned int dim = 2;
      const unsigned int deg_ = 13;
      const unsigned int num_pts = 2000;
      const unsigned int S = 10;
      std::vector<double> weights(dim,1);
      FMCA::MultiIndexSet<FMCA::WeightedTotalDegree> idcs_;
      idcs_.init(dim, deg_, weights);


      FMCA::HaltonSet<S> halton_pts(dim); //The S is the row of the resulting points
      Eigen::MatrixXd P (num_pts,dim);
        for (auto i = 0; i < P.rows(); ++i) {
        P.row(i) = halton_pts.EigenHaltonVector();
        halton_pts.next();
        }


        Eigen::Matrix<double,dim,num_pts> Tr = P.transpose();
        Eigen::MatrixXd V_init (num_pts, idcs_.index_set().size());
        Eigen::MatrixXd TD_xi_(dim,num_pts);
        Eigen::MatrixXd VD (num_pts,deg_+1);

        for (auto i = 0; i < num_pts; ++i){
          V_init.row(i) = evalPolynomials(Tr.col(i)).transpose();
      };
      
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

      
      Eigen::MatrixXd p2 (idcs_.index_set().size(), dim);
      for (int i = 0; i < idcs_.index_set().size(); ++i){
        for (int j = 0; j < dim; ++j)
          p2(i,j) = P(IND(i),j);
      };


      Eigen::MatrixXd V (idcs_.index_set().size(), idcs_.index_set().size());
      for (int i = 0; i < idcs_.index_set().size(); ++i){
        V.row(i) = V_init.row(IND(i));
      };

        // std::cout << V1.row(0).size() << std::endl;
        freopen( "file.txt", "w", stdout );
        cout <<  p2; // goes to file.txt

  return 0;
}
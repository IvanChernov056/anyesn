#ifndef     TYPSE_H
#define     TYPSE_H


#include    <armadillo>

using   Matrix = arma::mat;
using   SpMatrix = arma::sp_mat;
using   Column = arma::colvec;
using   Row    = arma::rowvec;

using   Data   = Matrix;


#include    <vector>

using   DoubleContainer = std::vector<double>;

#endif
#ifndef     NN_TYPSE_H
#define     NN_TYPSE_H


#include    <armadillo>

using   Matrix = arma::mat;
using   SpMatrix = arma::sp_mat;
using   Column = arma::colvec;
using   Row    = arma::rowvec;

using   Data   = Matrix;


#include    <vector>

using   DoubleContainer = std::vector<double>;
using   ColumnContainer = std::vector<Column>;



#include    <utility>

using   DataSet = std::pair<Data, Data>;


using   Activation = double (*) (double );

#endif
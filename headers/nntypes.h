#ifndef     NN_TYPSE_H
#define     NN_TYPSE_H


#include    <armadillo>
#include    <vector>
#include    <utility>



using   Matrix = arma::mat;
using   SpMatrix = arma::sp_mat;
using   Column = arma::colvec;
using   Row    = arma::rowvec;


using   SingleVector    = Column;
using   SingleData      = std::vector<SingleVector>;
using   MultipleVector  = std::vector<Column>;
using   MultipleData    = std::vector<MultipleVector>;
using   MultipleWeight  = std::vector<Matrix>;

using   DoubleContainer = std::vector<double>;
using   ColumnContainer = std::vector<Column>;



using   DataSet = std::pair<MultipleData, SingleData>;


using   Activation = double (*) (double );



#endif
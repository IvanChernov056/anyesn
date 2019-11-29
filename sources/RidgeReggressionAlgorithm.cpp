#include    "RidgeRegressionAlgorithm.h"


namespace nn {

    RidgeRegressionAlgorithm::RidgeRegressionAlgorithm (const DataSet& i_dataSet, double i_ridge)
        :  d_ridge(i_ridge), d_dataSet(i_dataSet)    
    {    
    }

    void RidgeRegressionAlgorithm::start(MultipleWeight& o_weights, Column& o_bias, const Column& i_totalIncomingSignal, BasicUnit *i_unit) {
        if (d_dataSet.first.empty() || d_dataSet.second.empty())
            throw std::runtime_error("RidgeRegressionAlgorithm::start -> empty dataSet");

        Matrix  matrixY = fn::makeMatrixFromContainer(d_dataSet.second);
        ColumnContainer conteinerX;
        
        for (auto& multiVec : d_dataSet.first)
            conteinerX.push_back(fn::uniteMultipeVector(multiVec));
        
        Matrix  matrixX = fn::makeMatrixFromContainer(conteinerX);
        Matrix  matrixS = CONCATINATE(vert, matrixX, MathVector(Row, ones, matrixX.n_cols));

        Matrix  matrixW = matrixY*matrixS.t()*INV_SYMPD(matrixS*matrixS.t() + d_ridge*EYE(matrixS.n_rows));

        fn::splitMatrixToMuliple(o_weights, matrixW, d_dataSet.first[0]);
        o_bias = matrixW.col(matrixW.n_cols-1);
    }

}
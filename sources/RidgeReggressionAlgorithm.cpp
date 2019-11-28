#include    "RidgeRegressionAlgorithm.h"


namespace nn {

    RidgeRegressionAlgorithm::RidgeRegressionAlgorithm (const DataSet& i_learnSet, double i_ridge)
        : BasicLearnAlgorithm(i_learnSet), d_ridge(i_ridge)    
    {    
    }

    void RidgeRegressionAlgorithm::start(MultipleWeight& i_weight, Column& i_bias, const Column& i_activation, BasicUnit *i_unit) {
        Matrix  matrixY = fn::makeMatrixFromContainer(d_dataSet.second);
        ColumnContainer conteinerX;
        
        for (auto& multiVec : d_dataSet.first)
            conteinerX.push_back(fn::uniteMultipeVector(multiVec));
        
        Matrix  matrixX = fn::makeMatrixFromContainer(conteinerX);
        Matrix  matrixS = CONCATINATE(vert, matrixX, MathVector(Row, ones, matrixX.n_cols));

        Matrix  matrixW = matrixY*matrixS.t()*INV_SYMPD(matrixS*matrixS.t() + d_ridge*EYE(matrixS.n_rows));

        std::vector<int> sizes;
        for (auto& v : d_dataSet.first[0]) 
            sizes.push_back(v.size());
        
        INFO_LOG("W:\n" << matrixW);
        int leftPoint = 0;
        int rightPoint = 0;
        for (int i = 0; i < sizes.size(); ++i) {
            rightPoint += sizes[i];
            i_weight[i] = matrixW.cols(leftPoint, rightPoint-1);
            leftPoint += sizes[i];
        }
        i_bias = matrixW.col(matrixW.n_cols-1);

        for(auto& w : i_weight)
            INFO_LOG("mat :\n" << w);
        INFO_LOG("bias :\n" << i_bias);
    }

}
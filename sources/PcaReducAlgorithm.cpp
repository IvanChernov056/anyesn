#include    "PcaReduceAlgorithm.h"

namespace nn {
    PcaReducAlgorithm::PcaReducAlgorithm(const MultipleData& i_inputData, int i_outDimention) 
        : d_inputData(i_inputData), d_outDimention(i_outDimention)
    {
    }            

    void PcaReducAlgorithm::start(MultipleWeight& o_weights, Column& o_bias, const Column& i_totalIncomingSignal, BasicUnit *i_unit) {
        Matrix  inputMatrix = fn::makeMatrixFromMultipleData(d_inputData);
        inputMatrix = CONCATINATE(vert, inputMatrix, MathVector(Row, ones, inputMatrix.n_cols));
        Matrix  covMatrix = fn::makeCovarianceMatrix(inputMatrix);
        Column  eigenVal;
        Matrix  eigenVec;

        EIGEN_SYM(eigenVal, eigenVec, covMatrix);
        
        if (NORM2(eigenVal) == 0)
            throw std::runtime_error(
                "PcaReducAlgorithm::start -> input datas variance is zero, pca not pass");
        
        Matrix result = eigenVec.col(eigenVec.n_cols-1);
        for(int i = 1; i<d_outDimention; ++i)
            result = CONCATINATE(horiz, eigenVec.col(eigenVec.n_cols-1-i), result);

        result = result.t();
        fn::splitMatrixToMuliple(o_weights, result, d_inputData[0]);
        o_bias = result.col(result.n_cols-1);

    }
}
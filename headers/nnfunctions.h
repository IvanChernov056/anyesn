#ifndef     NN_FUNCTIONS_H
#define     NN_FUNCTIONS_H


#include    "nntypes.h"


namespace nn {
    namespace fn {
        Matrix    makeMatrixFromContainer(const ColumnContainer& i_container);
        Matrix    makeMatrixFromMultipleData(const MultipleData& i_multiData);
        Matrix    makeCovarianceMatrix (const Matrix& i_inpMat);
        Column    uniteMultipeVector(const MultipleVector& i_input);
        void      bindSingleToMultiple(SingleData& i_singlesList, MultipleData& io_mulData);
        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const std::vector<int>& i_range);
        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const MultipleVector& i_rangeVector);
    }
}

#endif
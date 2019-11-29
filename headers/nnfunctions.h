#ifndef     NN_FUNCTIONS_H
#define     NN_FUNCTIONS_H


#include    "nntypes.h"


namespace nn {
    namespace fn {
        Matrix    makeMatrixFromContainer(const ColumnContainer& i_container);
        Column    uniteMultipeVector(const MultipleVector& i_input);
        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const std::vector<int>& i_range);
        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const MultipleVector& i_rangeVector);
    }
}

#endif
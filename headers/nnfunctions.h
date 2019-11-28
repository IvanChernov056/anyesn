#ifndef     NN_FUNCTIONS_H
#define     NN_FUNCTIONS_H


#include    "nntypes.h"


namespace nn {
    namespace fn {
        Matrix    makeMatrixFromContainer(const ColumnContainer& i_container);
        Column    uniteMultipeVector(const MultipleVector& i_input);
    }
}

#endif
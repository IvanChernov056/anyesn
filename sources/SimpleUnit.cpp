#include    "SimpleUnit.h"


namespace nn {

    SimpleUnit::SimpleUnit(int i_neuronsNumber, bool i_useBias, Activation i_func) :
        BasicForwardUnit(i_neuronsNumber), d_useBias(i_useBias), d_activFunc(i_func)     
    {
    }


    Column  SimpleUnit::operator()(const Column& i_input) {
        Column  result = d_useBias 
                    ? static_cast<Column>(d_weight*i_input + d_bias) 
                    : d_weight*i_input;
        return d_activFunc ? result.transform(d_activFunc) : result;
    }

    bool    SimpleUnit::fit(const Data& i_input, int i_epochCount) {
        int inpSize = i_input.n_rows;
        if (inpSize <= 0) {
            ERROR_LOG ("input data is empty");
            return false;
        }
        d_weight = RandnMatrix(d_neuronsNumber, inpSize);
        if (d_useBias) d_bias = RandnVector(Column, d_neuronsNumber);
        
        return true;
    }
}
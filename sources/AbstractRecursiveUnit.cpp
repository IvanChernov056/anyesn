#include    "AbstractRecursiveUnit.h"


namespace nn {

    Column AbstractRecursiveUnit::operator()(const Column& i_input, const Column& i_returned) {
        //there calculate d_state
        return i_input;
    }

    Column AbstractRecursiveUnit::operator()(const Column& i_input) {
        return this->operator()(i_input, d_state);
    }

}
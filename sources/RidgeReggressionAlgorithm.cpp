#include    "RidgeRegressionAlgorithm.h"


namespace nn {

    RidgeRegressionAlgorithm::RidgeRegressionAlgorithm (const DataSet& i_learnSet)
        : BasicLearnAlgorithm(i_learnSet)    
    {    
    }

    void RidgeRegressionAlgorithm::start(MultipleWeight& i_weight, Column& i_bias, const Column& i_activation, BasicUnit *i_unit) {
        INFO_LOG("THIS IS RIDGE\n" << i_bias << '\n' << i_activation);
        i_unit->forward(d_dataSet.first[0]);
        INFO_LOG("THIS IS RIDGE\n" << 2*i_activation);
    }

}